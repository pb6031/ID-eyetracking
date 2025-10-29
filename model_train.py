import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, accuracy_score, log_loss, confusion_matrix, precision_score, recall_score

# 이미지 로드 및 전처리
def load_and_process_images(folder_path, target_size=(128, 128)):
    #파일명 형식: [이름]_[label]_[id]_[문제번호]_[task이름].png
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            parts = filename.split('_')
            label = int(parts[1])
            id_val = int(parts[2])
            problem_number = int(parts[3])
            topic = parts[4].split('.')[0]
            # 이미지 로드
            img_path = os.path.join(folder_path, filename)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img) / 255.0
            
            data.append({'id': id_val,
                         'label': label,
                         'problem_number': problem_number,
                         'topic': topic,
                         'image': img_array})
            
    return pd.DataFrame(data)

# 이미지 데이터 경로
folder_path = './image_data/task'
image_data = load_and_process_images(folder_path)

pivoted = image_data.pivot_table(
    index=['id', 'problem_number'],
    columns='topic',
    values='image',
    aggfunc='first'
).dropna()

X_task1 = np.stack(pivoted['언어이해'].values)
X_task2 = np.stack(pivoted['유동추론'].values)
X_task3 = np.stack(pivoted['작업기억'].values)

labels_df = image_data.drop_duplicates('id').set_index('id')

sample_ids = pivoted.index.get_level_values(0).values
y = np.array([labels_df.loc[i, 'label'] for i in sample_ids])

# id별로 그룹화
group = sample_ids

def create_model():
    input_shape = (128, 128, 3)
    input1 = Input(shape=input_shape, name="Input_Task1")
    input2 = Input(shape=input_shape, name="Input_Task2")
    input3 = Input(shape=input_shape, name="Input_Task3")
    
    def cnn(input_layer):
        conv1 = Conv2D(16, kernel_size=(3, 3), activation='relu')(input_layer)
        maxp1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu')(maxp1)
        maxp2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(128, kernel_size=(3, 3), activation='relu')(maxp2)
        maxp3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        flatten = Flatten()(maxp3)
        return flatten
    
    # CNN 적용
    latent_task1 = cnn(input1)
    latent_task2 = cnn(input2)
    latent_task3 = cnn(input3)

    #task별 output
    output1 = Dense(1, activation='sigmoid', name="Output_Task1")(latent_task1)
    output2 = Dense(1, activation='sigmoid', name="Output_Task2")(latent_task2)
    output3 = Dense(1, activation='sigmoid', name="Output_Task3")(latent_task3)

    #latent vector를 concat
    concat= Concatenate()([latent_task1, latent_task2, latent_task3])
    dense = Dense(256, activation='relu')(concat)
    dropout = Dropout(0.3)(dense)
    
    #concat한 output
    output = Dense(1, activation='sigmoid', name='Output')(dropout)

    # 모델 정의 (task 3개+total)
    model = Model(inputs=[input1, input2, input3], outputs=[output1, output2, output3, output])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


os.makedirs('./model_kf/', exist_ok=True)

#Groupfold(4)
kf = GroupKFold(n_splits=4)

all_results = []
pred_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_task1, y, groups=group)):
    print(f"\nFold {fold + 1}")

    X1_train, X1_val = X_task1[train_idx], X_task1[val_idx]
    X2_train, X2_val = X_task2[train_idx], X_task2[val_idx]
    X3_train, X3_val = X_task3[train_idx], X_task3[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_model()

    checkpoint_path = f'./model_kf/cnn_fold_{fold + 1}.h5'
    mc = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=8, mode='min', verbose=1)
    reLR = ReduceLROnPlateau(patience=4, factor=0.5, verbose=1)
    
    # 모델 학습
    history = model.fit([X1_train, X2_train, X3_train], 
                    [y_train, y_train, y_train, y_train],
                    validation_data=([X1_val, X2_val, X3_val], [y_val, y_val, y_val, y_val]),
                    epochs=10,
                    batch_size=32,
                    callbacks=[es, mc, reLR],
                    verbose=1)

    model.load_weights(checkpoint_path)
    
    # 모델 평가    
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred_binary_total).ravel()

    acc = accuracy_score(y_val, y_pred_binary_total)
    f1 = f1_score(y_val, y_pred_binary_total)
    loss = log_loss(y_val, preds_total)
    precision = precision_score(y_val, y_pred_binary_total)
    recall = recall_score(y_val, y_pred_binary_total)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    all_results.append({
        'fold': fold + 1,
        'accuracy_total': acc,
        'loss_total': loss,
        'f1_score_total': f1,
        'precision_score_total': precision,
        'recall_score_total': recall,
        'specificity_total': specificity
    })

    # 개별 Task별 예측값
    y_pred_task1, y_pred_task2, y_pred_task3, y_pred_total = model.predict([X1_val, X2_val, X3_val])

    preds_task1 = y_pred_task1.flatten()
    preds_task2 = y_pred_task2.flatten()
    preds_task3 = y_pred_task3.flatten()
    preds_total = y_pred_total.flatten()

    y_pred_binary_task1 = (preds_task1 > 0.5).astype(int)
    y_pred_binary_task2 = (preds_task2 > 0.5).astype(int)
    y_pred_binary_task3 = (preds_task3 > 0.5).astype(int)
    y_pred_binary_total = (preds_total > 0.5).astype(int)

    #prediction csv 생성
    index_df = pivoted.reset_index().iloc[val_idx][['id', 'problem_number']].copy()
    index_df['True Label'] = y_val
    
    #task 마다 예측값 구해서 합치기
    index_df['0_Predicted Value'] = preds_task1
    index_df['0_Predicted Label'] = y_pred_binary_task1

    index_df['1_Predicted Value'] = preds_task2
    index_df['1_Predicted Label'] = y_pred_binary_task2

    index_df['2_Predicted Value'] = preds_task3
    index_df['2_Predicted Label'] = y_pred_binary_task3

    index_df['Total_Predicted Value'] = preds_total
    index_df['Total_Predicted Label'] = y_pred_binary_total

    pred_list.append(index_df)

