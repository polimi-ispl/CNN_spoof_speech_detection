from data_generator import *
from model import *
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import sklearn.model_selection

epochs = 40

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
print(gpus)


if __name__=='__main__':
    # Specify classification type
    classification_type = 'binary'
    traindev_eval = True
    balance_dataset = False
    upsample_dataset = True
    learning_rate = 0.00005

    train_batch_size = 32
    eval_batch_size = 1000

    if traindev_eval:
        train_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
        dev_classes_list = ['-', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06']
        classes_list = train_classes_list


        if classification_type == 'multiclass':
            n_classes = len(train_classes_list)
        else:
            n_classes = 2

        model = get_cnn_model_vggish_input(input_shape=(96, 64, 1), out_dim = n_classes)
        model.summary()

        train_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
        train_feat_root = "/nas/home/cborrelli/cnn_bot/features/logmelspectr/train"
        df_train = pd.read_csv(train_txt_path, sep=" ", header=None)
        df_train.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
        df_train = df_train.drop(columns="null")

        dev_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
        dev_feat_root = "/nas/home/cborrelli/cnn_bot/features/logmelspectr/dev"
        df_dev = pd.read_csv(dev_txt_path, sep=" ", header=None)
        df_dev.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
        df_dev = df_dev.drop(columns="null")

        if balance_dataset and classification_type == 'binary':
            num_bonafide = len(df_train[df_train['label'] == 'bonafide'])
            num_spoof_per_class = num_bonafide // len(train_classes_list)
            for c in train_classes_list:
                if c == '-':
                    continue
                total_samples_per_class_train = len(df_train[df_train['system_id'] == c])
                to_drop_train = df_train[df_train['system_id'] == c].sample( total_samples_per_class_train - num_spoof_per_class,
                                                                       random_state=2)
                df_train.drop(to_drop_train.index, inplace=True)
            df_train.reset_index(inplace=True)
            df_dev.reset_index(inplace=True)

        if upsample_dataset and classification_type == 'binary':
            num_spoof = len(df_train[df_train['label'] == 'spoof'])
            num_bonafide = len(df_train[df_train['label'] == 'bonafide'])

            replicas = int(np.ceil(num_spoof / num_bonafide))

            temp_df = []
            count_bonafide = 0
            for row in df_train.itertuples(index=False):
                if row.label == 'bonafide':
                    temp_df.extend([list(row)] * replicas)
                    count_bonafide += 1
                else:
                    temp_df.append(list(row))

            df_train = pd.DataFrame(temp_df, columns=df_train.columns)
            df_train.reset_index(inplace=True)


        train_generator = VGGishDataGenerator(dataframe=df_train,
                                              feature_path=train_feat_root,
                                              batch_size=train_batch_size,
                                          classification_type=classification_type,
                                          classes_list=train_classes_list)
        dev_generator = VGGishDataGenerator(dataframe=df_dev,
                                            feature_path=dev_feat_root,
                                            batch_size=eval_batch_size,
                                        classification_type=classification_type,
                                        classes_list=dev_classes_list,
                                            shuffle=False)
    else:
        eval_classes_list = ['-', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 'A14',
                             'A15', 'A16', 'A17', 'A18', 'A19']
        classes_list = eval_classes_list

        if classification_type == 'multiclass':
            n_classes = len(eval_classes_list)
        else:
            n_classes = 2

        model = get_cnn_model_vggish_input(input_shape=(96, 64, 1), out_dim=n_classes)
        model.summary()

        eval_txt_path = "/nas/public/dataset/asvspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
        eval_feat_root = "/nas/home/cborrelli/cnn_bot/features/logmelspectr/eval"
        df_eval = pd.read_csv(eval_txt_path, sep=" ", header=None)
        df_eval.columns = ["speaker_id", "audio_filename", "null", "system_id", "label"]
        df_eval = df_eval.drop(columns="null")

        # this if is just a backup, binary is only on train classes
        if balance_dataset and classification_type == 'binary':
            num_bonafide = len(df_eval[df_eval['label'] == 'bonafide'])
            num_spoof_per_class = num_bonafide // len(eval_classes_list)
            for c in eval_classes_list:
                if c == '-':
                    continue
                total_samples_per_class_eval = len(df_eval[df_eval['system_id'] == c])
                to_drop_eval = df_eval[df_eval['system_id'] == c].sample(
                    total_samples_per_class_eval - num_spoof_per_class,
                    random_state=2)
                df_eval.drop(to_drop_eval.index, inplace=True)

            df_eval.reset_index(inplace=True)

        # this if is just a backup, binary is only on train classes
        if upsample_dataset and classification_type == 'binary':
            num_spoof = len(df_eval[df_eval['label'] == 'spoof'])
            num_bonafide = len(df_eval[df_eval['label'] == 'bonafide'])

            replicas = num_spoof // num_bonafide

            temp_df = []
            for row in df_eval.itertuples(index=False):
                if row['label'] == 'bonafide':
                    temp_df.extend([list(row)] * replicas)
                else:
                    temp_df.append(list(row))

            df_eval = pd.DataFrame(temp_df, columns=df_eval.columns)
            df_eval.reset_index(inplace=True)

        df_eval_train, df_eval_test = sklearn.model_selection.train_test_split(df_eval, test_size=0.2, random_state=2)
        df_eval_train.reset_index(inplace=True, drop=True)
        df_eval_test.reset_index(inplace=True, drop=True)

        train_generator = VGGishDataGenerator(dataframe=df_eval_train,
                                              feature_path=eval_feat_root,
                                              batch_size=train_batch_size,
                                              classes_list=eval_classes_list,
                                              classification_type=classification_type,
                                              traindev_eval=traindev_eval)
        dev_generator = VGGishDataGenerator(dataframe=df_eval_test,
                                            feature_path=eval_feat_root,
                                            batch_size=eval_batch_size,
                                            classes_list=eval_classes_list,
                                            classification_type=classification_type,
                                            traindev_eval=traindev_eval)


    checkpoint_path = '/nas/home/cborrelli/cnn_bot/checkpoints/simple_cnn_vggish_input/model_classification_{}_classes_{}_lr_{}_upsample_{}'.format(
        classification_type, '_'.join(classes_list), learning_rate, upsample_dataset)
    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5),
        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss'),
    ]
    ## add callback
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    if classification_type == 'binary':
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            #tf.keras.metrics.TruePositives(name='tp'),
            #tf.keras.metrics.FalsePositives(name='fp'),
            #tf.keras.metrics.TrueNegatives(name='tn'),
            #tf.keras.metrics.FalseNegatives(name='fn'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            # tf.keras.metrics.Precision(name='precision'),
            # tf.keras.metrics.Recall(name='recall'),
            # tf.keras.metrics.AUC(name='auc'),
        ]
    if classification_type == 'multiclass':
        loss = tf.keras.losses.CategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        ]

    model.compile(loss=loss, optimizer=opt, metrics=metrics, weighted_metrics=['accuracy'])
    history = model.fit(train_generator, validation_data=dev_generator,
                        epochs=epochs, callbacks=my_callbacks)
                        #class_weight=class_weight)

    history_name = '/nas/home/cborrelli/cnn_bot/history/simple_cnn_vggish_input/model_classification_{}_classes_{}_lr_{}_upsample_{}.npy'.format(
        classification_type, '_'.join(classes_list), learning_rate, upsample_dataset)
    # Save history
    np.save(history_name, history.history)