import json
import sys
import os
# Other imports

def main():
    """
    Entry point to the training application

    :rtype: None
    :return: None
    """
    opts = json.loads(sys.argv[1])
    py_dir = os.path.dirname(os.path.realpath(__file__))

    # Read config
    config_file = os.path.join(py_dir, "path/to/config.json")
    with open(config_file, "r") as cf:
        config = json.load(cf)
        
    # Create model directory
    driver = Driver(model_id=opts["model_id"],
                    base_dir=opts["model_dir"],
                    template_dir=os.path.join(py_dir, "path/to/templates"),
                    common_dir=os.path.join(py_dir, "path/to/common"),
                    config_dir=os.path.join(py_dir, "path/to/config"))
    driver.setup()

    # Create data frame for training by processing the input data
    preprocessor = Preprocessor(config=config, opts=opts)
    df_train, preprocessing_info = preprocessor.run()

    # Training
    trainer = Trainer(model_id=opts["model_id"],
                      dataframe=df_train,
                      model_dir=driver.model_dir,
                      config=config)

    algo = opts["model_type"]
    training_info = trainer.run(algo)

    # Assembly
    info = driver.make_model(preprocessing_info, training_info)

    # Cleanup
    driver.cleanup()
    ut.print_progress(opts["model_id"], 100)

    # Print model information
    info.update({"training_lots": preprocessing_info["training_lots"]})
    print json.dumps(info)