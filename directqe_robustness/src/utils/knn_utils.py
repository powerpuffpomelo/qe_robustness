import faiss
import torch


def faiss_search(train_dataset_faiss_path,
                    test_feature,
                    k=10):
    """
    读取训练集特征，并从中检索与测试样本最相似的训练样本
    k是检索数量
    """
    faiss_index = faiss.read_index(train_dataset_faiss_path)
    KNN_distances, KNN_indices = faiss_index.search(test_feature, k=10)
    
    return KNN_distances, KNN_indices


def write_knn_result_sent(knn_indices,
                        train_dataset,
                        test_id,
                        test_dataset,
                        log_path,):
    """
    展示knn检索到的结果（句子级）
    先记录该测试样本，再根据检索到的knn_indices依次记录训练样本
    """
    #print("knn_indices==================")
    #print(KNN_indices)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('---------------------------test_id: ' + str(test_id) + '---------------------------\n')
        for ii, test_sample_i in enumerate(test_dataset.showitem(test_id)):
            if ii == 3: f.write("gold hter: ")
            elif ii == 4: continue
            elif ii == 5: f.write("pred hter: ")
            f.write(test_sample_i)
        f.write("\n=================================================\n\n")
        f.write(' '.join([str(id) for id in knn_indices[0]]) + '\n')
        for id in knn_indices[0]:
            knn_sample = train_dataset.showitem(id)
            #print(knn_sample)
            f.write('---------------------------train id: ' + str(id) + '---------------------------\n')
            for ii,sample_i in enumerate(knn_sample):
                if ii == 3: f.write("gold hter: ")
                elif ii == 4: continue
                elif ii == 5: f.write("pred hter: ")
                f.write(sample_i)