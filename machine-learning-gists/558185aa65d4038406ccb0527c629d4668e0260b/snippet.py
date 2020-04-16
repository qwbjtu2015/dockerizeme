# !/usr/bin/env python
#  -*- coding: utf-8 -*-
#
# ブログ記事「オンライン広告をどのようにプロモーションする？ - 劣モジュラ性と局所探索で解決」に記載した内容のサンプル実装
# https://www.altus5.io/blog/machine-learning/2017/01/06/submodular2/

# 商品の定価
PRICE = 2000

# 全ユーザー
users = {}


def add_user(name, followers):
    # サンプルデータを登録する関数
    users[name] = {
        'followers': followers,
        'followings': set()
    }


add_user('A', ['B', 'I', 'G', 'E', 'F'])
add_user('B', ['I', 'G', 'A', 'F'])
add_user('C', ['D'])
add_user('D', ['C'])
add_user('E', ['D', 'G', 'A', 'F'])
add_user('F', ['A', 'B', 'E', 'J'])
add_user('G', ['E', 'D', 'H'])
add_user('H', ['I'])
add_user('I', ['H', 'K'])
add_user('J', ['F', 'K'])
add_user('K', ['I', 'L'])
add_user('L', ['K'])

# followersからfollowingsを作成する
for name, user in users.items():
    for follower_name in user['followers']:
        follower = users[follower_name]
        follower['followings'].add(name)
for name, user in users.items():
    pass

# 下記はツイート人数に対応する商品の購入確率を返す配列
# （添字はフォローしている商品購入ユーザー数）
# サンプルなので、仮の購入確率をおく
# 本来は実績値に基づき統計的に決める
PURCHASE_PROB = [0.005, 0.010, 0.0115, 0.0118, 0.01185, 0.0119]


def campaign_obj():
    # キャンペーン対象者を決める関数

    # 対象ユーザー (返り値、最初は空)
    target_users = []
    round = 1
    while True:
        print('\n------------%dth round------------' % round)
        user_with_max_benefit = None
        max_benefit = 0.000
        # キャンペーン対象候補(candidate_users) 
        # フォローワー数の多い順に並べ替えてから、実行する
        candidate_users = set(users)-set(target_users)
        candidate_users = sorted(candidate_users,
            key=lambda x: str(len(users[x]['followers']))+'_'+x,
            reverse=True)
        print('candidate_users: '+str(candidate_users))
        for i in candidate_users:
            if round == 1:
                print('When adding %s into empty set' % i)
            else:
                print('When adding %s into %s' 
                    % (i, sorted(target_users)))
            revenue = 0.000
            for j in set(users[i]['followers'])-set(target_users):
                new_target_users = set(target_users) | set([i])
                followed_by_j_before = \
                    set(target_users) & set(users[j]['followings'])
                followed_by_j_after = \
                    new_target_users & set(users[j]['followings'])
                # iを加える前のjの購入確率
                prob_before = PURCHASE_PROB[len(followed_by_j_before)]
                # iを加えた後のjの購入確率
                prob_after = PURCHASE_PROB[len(followed_by_j_after)]
                # jの購入確率の増分
                prob_increase = prob_after - prob_before
                revenue += PRICE * prob_increase
            set(target_users) & set(users[i]['followings'])
            followed_by_i = set(target_users) & set(users[i]['followings'])
            # 機会損失額(cost) = 定価 * 定価でも買ってくれた確率
            # 便益(benefit)は、利益から機会損失額を引いて求める
            cost = PRICE * PURCHASE_PROB[len(followed_by_i)]
            benefit = revenue - cost
            print('revenue = %3.3f, ' % revenue),
            print('rcost = %3.3f, ' % cost),
            print('benefit = %3.3f.' % benefit)
            if benefit > max_benefit:
                # ユーザーiをuser_with_max_benefitにセット
                user_with_max_benefit = i
                max_benefit = benefit
                print('Updated user_with_max_benefit:'),
                print('%s' % user_with_max_benefit)

        if max_benefit > 0:
            if round == 1:
                print('Added %s into target_users empty set'
                    % user_with_max_benefit)
            else:
                print('Added %s into target_users %s'
                    % (user_with_max_benefit,
                       str(sorted(target_users))))
            target_users = \
                set(target_users) | set(user_with_max_benefit)
        else:
            break

        round += 1

    print('target_users = '+str(sorted(target_users)))
    return target_users

# 定義終わり

# 実行
campaign_obj()
