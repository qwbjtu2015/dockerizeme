import h2o

def h2o_train(model, X_train, y_train, X_val=None, y_val=None):
    print('h2o_train - training...')
    X_y_train = np.c_[ X_train, y_train ]
    frame_train = h2o.H2OFrame(X_y_train)
    
    if X_val is not None and y_val is not None:
        X_y_val = np.c_[ X_val, y_val ]
        frame_val = h2o.H2OFrame(X_y_val)
    else:
        frame_val = None
        
    if type(model) == h2o.automl.autoh2o.H2OAutoML:
        # it was necessary to assign frame_val to leaderboard_frame.
        # h2o documentation says validation_frame would be automatically used as leaderboard_frame,
        # but it didn't work.
        model.train(x=frame_train.columns[:-1], y=frame_train.columns[-1], training_frame=frame_train, 
                    validation_frame=frame_val, leaderboard_frame=frame_val)
    else:
        model.train(x=frame_train.columns[:-1], y=frame_train.columns[-1], training_frame=frame_train, 
                    validation_frame=frame_val)


def h2o_predict(model, X_test):
    print('h2o_predict - predicting...')
    frame_test = h2o.H2OFrame(X_test)
    pred = model.predict(frame_test)
    return pred.as_data_frame().values

def evaluate_model_h2o(X, y, model):
    y_pred = h2o_predict(model, X)
    mape1 = mape(y, y_pred)
    mse1 = mse(y, y_pred)
    result = { 'mse': mse1, 'mape': mape1, 'size': y_pred.shape[0] }
    return result

def h2o_not_default_params_df(model):
    params = model.get_params()
    result = []
    ignore_params = ['model_id', 'training_frame', 'validation_frame', 'response_column', 'ignored_columns']
    for k in params:
        if k not in ignore_params and params[k]['default_value'] != params[k]['actual_value']:
            result.append({ 'param': k, 'default_value': params[k]['default_value'], 
                            'actual_value': params[k]['actual_value']})
    df_ = pd.DataFrame(result)
    cols = ['param', 'default_value', 'actual_value']
    return df_[ cols ]

def h2o_not_default_params_str(model):
    params = model.get_params()
    str_ = ''
    ignore_params = ['model_id', 'training_frame', 'validation_frame', 'response_column', 'ignored_columns']
    start = True
    for k in params:
        if k not in ignore_params and params[k]['default_value'] != params[k]['actual_value']:
            if start:
                start = False
            else:
                str_ += ', '
            
            actual_value = params[k]['actual_value']
            
            try:
                if str(actual_value) in ['True', 'False']:
                    actual_value = bool(actual_value)
                elif int(actual_value) == float(actual_value):
                    actual_value = int(actual_value)
                else:
                    actual_value = float(actual_value)
                str_ += "{}={}".format(k, actual_value)
            except:
                str_ += "{}='{}'".format(k, actual_value)
    return str_

def h2o_print_leaderboard(lb_frame, top_n=999999):
    df = lb_frame.as_data_frame()
    for i in range(0, min(top_n, df.shape[0])):
        model_id = df['model_id'][i]
        print(df[i:i+1].to_string(index=False))
        best_model = h2o.get_model(model_id)
        pprint(h2o_not_default_params_str(best_model))
        print()