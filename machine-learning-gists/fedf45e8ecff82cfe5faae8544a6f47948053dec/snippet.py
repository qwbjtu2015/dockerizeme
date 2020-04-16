# -*- coding: utf-8 -*-
#
# Author: Taylor Smith
#
# This function provides an interface for splitting a sparse ratings
# matrix RDD into a train and test set for use in collaborative
# filtering in PySpark applications.
#
# Dependencies:
#   * scikit-learn >= 0.18
#   * numpy >= 1.11

from __future__ import absolute_import

from sklearn.utils.validation import check_random_state
import numpy as np

import random
import sys

__all__ = [
    'train_test_split'
]


def aggregate_values_by_key(rdd):
    """Given an RDD in the form of:

        ``RDD[(0, 1), (0, 2), (1, 2)]``

    Aggregate each key (first index in the tuple) into an RDD of 
    keys to lists of all values associated with the key.

    Examples
    --------
    >>> x = sc.parallelize([("a", 1), ("b", 1), ("a", 2)])
    >>> sorted(aggregate_values_by_key(x).collect())
    [('a', [1, 2]), ('b', [1])]

    Parameters
    ----------
    rdd : RDD
        The RDD in the form of ``RDD[(0, 1), (0, 2), (1, 2)]``.
    """
    def to_list(a):
        return [a]

    def append(a, b):
        a.append(b)
        return a

    def extend(a, b):
        a.extend(b)
        return a

    # group by (using aggregate) the key to get a list of values
    return rdd.map(lambda r: (r[0], r[1]))\
              .combineByKey(to_list, append, extend)


def train_test_split(rdd, train_size=0.8, seed=None):
    """Given a sparse matrix represented in an RDD in the format:

        ``RDD[(user, item, rating), ..., (user, item, rating)]``

    Generate a split that ensures that the train set contains only users and
    items that are also contained in the (full) testing set. This means only
    users and items which have multiple ratings are eligible for masking out
    of the training set.

    Train-test splits in collaborative filtering differ greatly from those in
    traditional machine learning domains, where the most complicated splits are
    stratified by a series of vectors. In collaborative filtering, there will
    be overlap between the training set and test sets, but we randomly select
    events to be omitted from the TRAIN set [1]. This is subject to the
    aforementioned constraint, however. Consider, for example the following
    sparse matrix of rating events in RDD form (using a list for simpler
    reading):

    >>> [(0, 0, 1.), (0, 1, 1.), (1, 0, 1.),
    ...  (1, 1, 1.), (2, 1, 1.), (3, 3, 1.)]

    The following would be a VALID train-test split of the input:

    >>> ([(0, 0, 1.), (1, 0, 1.), (2, 1, 1.), (3, 3, 1.)]),  # train
    ...  [(0, 0, 1.), (0, 1, 1.), (1, 0, 1.),
    ...   (1, 1, 1.), (2, 1, 1.), (3, 3, 1.)])  # test set

    This is because all items and users that appear in the test set also appear
    in the training set. It's the same concept as having a new factor level in
    the test set using scikit-learn. Things will break down. Therefore, the
    following is an INVALID train/test split:

    >>> ([(0, 0, 1.), (1, 0, 1.), (1, 1, 1.), (3, 3, 1.)]),  # train
    ...  [(0, 0, 1.), (0, 1, 1.), (1, 0, 1.),
    ...   (1, 1, 1.), (2, 1, 1.), (3, 3, 1.)])  # test set

    This is due to the fact that user 2 shows up in the test set,
    but not in the training set. The model won't be able to generate
    predictions for user 2. Thus, this train-test split procedure
    retains all records from the test set, and filters a percentage of 
    eligible events from the train set.

    Parameters
    ----------
    rdd : RDD
        The sparse matrix in RDD form:
        ``RDD[(user, item, rating), ..., (user, item, rating)]``

    train_size : float, optional (default=0.8)
        The size of the train set with respect to the input RDD.
        Note that the size will not be exact, as there is only a certain number
        of eligible users/items for the train set, but represents the 
        probabilistic down-sampling ratio. The actual number of records 
        retained may be slightly larger or smaller.

    seed : int or None, optional (default=None)
        The seed used to initialize the random state.

    Notes
    -----
    * Running this method requires a live Spark connection. If you try to 
      run it without a connection, you will get an import error.
      
    * While every operation in this method happens in a distributed 
      fashion, the unique item IDs are tracked in a set. Therefore, the
      IDs of the unique items must fit into memory (even for millions of
      items, this should not be an issue).

    Returns
    -------
    train : RDD
        The training set, a subset of the test set.

    test : RDD
        The testing set, equivalent to the input RDD.
        
    References
    ----------
    .. [1] Train/test Splits in Collaborative Filtering - http://bit.ly/2h2jXoJ
    """
    # if seed is undefined, default to sys time
    if seed is None:
        seed = random.randint(0, sys.maxint)

    # we need the input rdd to be cached so we don't accidentally recompute it
    # over and over again (test is equal to the input rdd)
    test = rdd.cache()

    def compute_train():
        # get users mapped to a list of the items they've rated. RDD will look
        # like RDD[(usr1, [itm1, itm4]), (usr2, [itm0, itm6]), ...]
        users_items = aggregate_values_by_key(
            test.map(lambda r: (r[0], r[1]))).cache()

        # function for getting unique items from an RDD in the above format
        def unq_items(x):
            return set(x.flatMap(lambda ur: ur[1]).distinct().collect())

        # get all the unique items so we can track when we've hit
        # them all. since it's just unique items, this will all fit
        # into a set in memory for O(1) lookups and set differencing
        still_needed_items = unq_items(users_items)

        def sample_at_ratio(ur, rs):
            usr, itm = ur  # unpack the tuple

            # make the items a numpy array, mask it
            itm = np.asarray(itm, dtype=np.long)
            mask = (rs.rand(len(itm)) < train_size)  # type: np.ndarray

            # we need to make sure at least ONE item/user is sampled or else we
            # won't get anything from this user/item, and that violates our
            # requirements!
            while not mask.any():
                mask = (rs.rand(len(itm)) < train_size)

            # now we know there are at least SOME items that are
            # retained per user
            return usr, itm[mask].tolist()

        def sample_partition_at_ratio(partition_index, partition):
            # this looks cryptic and funky, but if we use the same seed in
            # every partition, different executors will initialize the same
            # random state, and then ordering becomes non-random. This adds
            # a bit more randomness into the random_state initialization
            random_state = check_random_state(partition_index * seed)
            for ur in partition:
                yield sample_at_ratio(ur, random_state)

        # mask out the required ratio of items to probabilistically down-sample
        # the training set, and then identify which items we're missing
        train = users_items.mapPartitionsWithIndex(
            sample_partition_at_ratio).cache()

        # now we KNOW that all of the users have at least some item(s)
        # so all that matters is that we make sure all items are represented in
        # some capacity
        still_needed_items -= unq_items(train)

        # corner case 1: there are no more needed items -- everything
        # was sampled!
        if not still_needed_items:
            return train

        # if we get to this point, it means that every user has been sampled,
        # but there are some items that have not yet been sampled. So we'll do
        # something a little different... group each of the items needed by the
        # users that have rated it, and then probabilistically sample those
        # users similar to above such that the items are represented at the
        # ratio at at which they were rated. Now this RDD will resemble:
        # RDD[(itm3, [usr6, usr0]), (itm12, [usr4, usr1]), ...]
        needed_items_to_users = aggregate_values_by_key(
                        test.filter(lambda r: r[1] in still_needed_items)
                            .map(lambda r: (r[1], r[0])))  # map in reverse

        # apply the function above against the reversed RDD, and then flatMap
        # it out in reverse order prior to aggregating with the train RDD
        def swap_user_items(iu):
            item, users = iu  # unpack
            return [(user, [item]) for user in users]

        imputed_items = needed_items_to_users.mapPartitionsWithIndex(
            sample_partition_at_ratio).flatMap(swap_user_items).cache()

        # extend two lists together and return the left one
        def extend(list_a, list_b):
            # any users that were not selected for the left-out items will
            # have an empty value after the left join. thus, left list
            # shouldn't ever be None, but check it just for caution's sake...
            if list_a is None:
                return list_b
            if list_b is None:
                return list_a

            list_a.extend(list_b)
            return list_a

        # join the two together and merge the items lists together where
        # necessary. a LEFT join will suffice since we are CERTAIN that all
        # users are present in the train RDD
        return train.leftOuterJoin(imputed_items)\
                    .map(lambda uii: (uii[0], extend(uii[1][0], uii[1][1])))\
                    .cache()

    # compute the training set
    train_set = compute_train()

    # the train set looks like this:
    # RDD[(usr0, [itm1, itm6]), ...]
    # so, it consists of the user and the items to sample for that user in
    # order to satisfy the constraints while down-sampling as much as possible.
    # so NOW we have to join it back up with the input RDD (test) to get the
    # ratings unpacked, and then return

    # map the train set out so there are compound keys for a join:
    def unpack_user_items(ui):
        user, items = ui  # unpack the tuple

        # need the None value for the join
        return [((user, item), None) for item in items]

    train_pre_join = train_set.flatMap(unpack_user_items)

    # now map the test set with the rating as the value so we can
    # get the rating out of the join
    test_pre_join = test.map(lambda uir: ((uir[0], uir[1]), uir[2]))

    # do the join (inner!) and cache it
    def reform_ratings(urr):
        (user, item), (_, rating) = urr  # unpack ((user, item), (None, rtg))
        return user, item, rating

    train_set = train_pre_join.join(test_pre_join).map(reform_ratings).cache()
    return train_set, test


train_test_split.__test__ = False  # avoid problem with nose
