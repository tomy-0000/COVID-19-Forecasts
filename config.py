dataset_config = {"seq": 14,
                  "val_test_len": 30,
                  "batch_size": 10000}
epoch = 30000
patience = 500
repeat_num = 2

net1_config = {"hidden_size": 32,
               "num_layers": 1}

net2_config = {"hidden_size": 32,
               "num_layers": 1}

net3_config = {"weather_embedding_dim": 8,
               "hidden_size": 32,
               "num_layers": 1}