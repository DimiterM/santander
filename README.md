Дипломна работа
Препоръчваща система, основана на историята на поръчките и демографските данни на потребителите
Димитър Пенчев Мутафчиев, 24990

Необходими ресурси:
Python (64-bit)   3.6.0       https://www.python.org/
Keras             2.0.3       https://keras.io/
matplotlib        2.0.0       https://matplotlib.org/
numpy             1.12.1+mkl  http://www.numpy.org/
pandas            0.19.0      http://pandas.pydata.org/
scikit-learn      0.18        http://scikit-learn.org/stable/
tensorflow        1.0.1       https://www.tensorflow.org/
h5py              2.7.0       http://www.h5py.org/
Инсталират се с командата `pip install`
(при проблеми на Windows OS: някои могат да бъдат изтеглени от тук: http://www.lfd.uci.edu/~gohlke/pythonlibs/)

Множество от данни: линк за изтегляне https://www.kaggle.com/c/santander-product-recommendation/data


main.py [-h] [--trainset TRAINSET] [--testset TESTSET] [-m TRAIN_MONTH]
        [-t TEST_MONTH] [--use_buckets] [-f MODEL_FILENAME]
        [-a RNN_ARCHITECTURE] [-g GO_DIRECTION] [-n NUM_EPOCHS]
        [-b BATCH_SIZE] [-l LEARNING_RATE]
опционални аргументи:
  -h, --help                                                ### изписва това съобщение и завършва
  --trainset TRAINSET                                       ### име на файл с тренировъчни данни
  --testset TESTSET                                         ### име на файл с тестови данни
  -m TRAIN_MONTH, --train_month TRAIN_MONTH                 ### номер (1-17) на последния месец за тренировъчните данни
  -t TEST_MONTH, --test_month TEST_MONTH                    ### номер (2-18) на месец за тестовите данни
  --use_buckets                                             ### флаг, който указва да не се използва маскиране на липсващата история, а да се направи списък от тренировъчни данни с различни дължини (по подразбиране флагът не се използва)
  -f MODEL_FILENAME, --model_filename MODEL_FILENAME        ### файл с готов предварително обучен модел
  -a RNN_ARCHITECTURE, --rnn_architecture RNN_ARCHITECTURE  ### рекурентна архитектура, която да се използва (по подразбиране: LSTM)
  -g GO_DIRECTION, --go_direction GO_DIRECTION              ### посока на четене на историята: 1 = в хронологичен ред (по подразбиране), -1 = наобратно, 2 = двупосочно
  -n NUM_EPOCHS, --num_epochs NUM_EPOCHS                    ### брой епохи за обучение (по подразбиране: 20)
  -b BATCH_SIZE, --batch_size BATCH_SIZE                    ### размер на партидите (по подразбиране: 256)
  -l LEARNING_RATE, --learning_rate LEARNING_RATE           ### скорост на обучение (по подразбиране: 0.001)


make_dataframe.py [-h] [--noimpute] [--test]
опционални аргументи:
  -h, --help  ### изписва това съобщение и завършва
  --noimpute  ### флаг, който указва дали да не се попълват липсващите стойности (по подразбиране: попълват се)
  --test      ### флаг, който указва дали да се преобразува тестовото или тренировъчното множество


make_submission.py [-h] results_filename submit_filename
позиционни аргументи:
  results_filename  ### име на файл с резултати от класификацията
  submit_filename   ### име на файл, в който да бъдат записани списъците с препоръки
опционални аргументи:
  -h, --help        ### изписва това съобщение и завършва
  -d DATASET, --dataset DATASET   ### файл с последни данни за клиентите (по подразбиране: "df.csv")
  -m MONTH, --month MONTH         ### последен месец (по подразбиране: 17)


activations_y_plots.py [-h] [--trainset TRAINSET] [--testset TESTSET]
                       [-l HIDDEN_LAYER_INDEX] [-m TRAIN_MONTH]
                       [-t TEST_MONTH] [-n TOP_N_SAMPLES]
                       model_filename
позиционни аргументи:
  model_filename                                                    ### файл с готов предварително обучен модел
опционални аргументи:
  -h, --help                                                        ### изписва това съобщение и завършва
  --trainset TRAINSET                                               ### име на файл с тренировъчни данни
  --testset TESTSET                                                 ### име на файл с тестови данни
  -l HIDDEN_LAYER_INDEX, --hidden_layer_index HIDDEN_LAYER_INDEX    ### индекс на слой от модела (по подразбиране: -5, което е конкатениращият слой)
  -m TRAIN_MONTH, --train_month TRAIN_MONTH                         ### номер (1-17) на последния месец за тренировъчните данни
  -t TEST_MONTH, --test_month TEST_MONTH                            ### номер (2-18) на месец за тестовите данни
  -n TOP_N_SAMPLES, --top_n_samples TOP_N_SAMPLES                   ### да се покажат толкова на брой примери, които активират максимално всеки неврон в скрития слой (по подразбиране: 10)
  -s SHUFFLE_SPLIT_SIZE, --shuffle_split_size SHUFFLE_SPLIT_SIZE    ### дял от всички данни, които да се изобразят (по подразбиране: 0.005)


