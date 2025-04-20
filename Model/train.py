#!/usr/bin/env python
# coding: utf-8

# In[1]:

# 设置内联绘图
# get_ipython().run_line_magic('matplotlib', 'inline')
import copy
from time import time
import matplotlib.pyplot as plt
import seaborn as sns  # 新增导入

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from config import BIN_config_DBPE
from models import BIN_Interaction_Flat
from stream import BIN_Data_Encoder

import copy
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

# 设置随机种子
torch.manual_seed(2)
np.random.seed(3)

# 配置设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")


# In[2]:


# 直接在 Notebook 中硬编码参数，避免 argparse 冲突
class Args:
    batch_size = 32     # 原 -b/--batch-size
    workers = 20        # 原 -j/--workers
    epochs = 1         # 原 --epochs
    task = "biosnap"    # 原 --task
    lr = 1e-4           # 原 --lr

args = Args()


# In[3]:


def get_task(task_name):
    if task_name.lower() == 'biosnap':
        return './dataset/BIOSNAP/full_data'
    elif task_name.lower() == 'bindingdb':
        return './dataset/BindingDB'
    elif task_name.lower() == 'davis':
        return './dataset/DAVIS'

def test(data_generator, model, threshold=None):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, p, d_mask, p_mask, label) in enumerate(data_generator):
        # 将数据移动到选定的设备
        d = d.long().to(device)
        p = p.long().to(device)
        d_mask = d_mask.long().to(device)
        p_mask = p_mask.long().to(device)
        label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
        
        with torch.no_grad():
            score = model(d, p, d_mask, p_mask)
            
        m = torch.nn.Sigmoid()
        logits = torch.squeeze(m(score))
        loss_fct = torch.nn.BCELoss()
        loss = loss_fct(logits, label)
        
        loss_accumulate += loss
        count += 1
        
        logits = logits.detach().cpu().numpy()
        label_ids = label.to('cpu').numpy()
        y_label += label_ids.flatten().tolist()
        y_pred += logits.flatten().tolist()
    
    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)
    precision = tpr / (tpr + fpr + 1e-10)
    f1 = 2 * precision * tpr / (tpr + precision + 1e-10)
    thred_optim = thresholds[5:][np.argmax(f1[5:])] if len(thresholds) > 5 else 0.5

    # 使用传入的阈值或自动计算的阈值
    if threshold is None:
        threshold = thred_optim

    y_pred_labels = (np.asarray(y_pred) >= threshold).astype(int)
    auc_k = roc_auc_score(y_label, y_pred)
    auprc = average_precision_score(y_label, y_pred)
    f1_val = f1_score(y_label, y_pred_labels)

    return auc_k, auprc, f1_val, y_pred, loss, thred_optim, y_label



# In[4]:



def main():
    from config import BIN_config_DBPE
    from models import BIN_Interaction_Flat
    from stream import BIN_Data_Encoder

    config = BIN_config_DBPE()
    config['batch_size'] = args.batch_size

    # 初始化指标存储
    train_losses = []
    val_aucs = []
    val_auprcs = []
    val_f1s = []
    val_losses = []

    model = BIN_Interaction_Flat(**config).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 数据加载
    dataFolder = get_task(args.task)
    # 添加数据文件存在性检查
    for file in ['train.csv', 'val.csv', 'test.csv']:
        if not os.path.exists(os.path.join(os.path.abspath(str(dataFolder)), file)):
            raise FileNotFoundError(f"数据文件 {file} 在路径 {dataFolder} 中不存在")
    # df_train = pd.read_csv(dataFolder + '/train.csv')
    # df_val = pd.read_csv(dataFolder + '/val.csv')
    # df_test = pd.read_csv(dataFolder + '/test.csv')
    # 添加数据读取错误处理
    try:
        df_train = pd.read_csv(os.path.join(dataFolder, 'train.csv'))
        df_val = pd.read_csv(os.path.join(dataFolder, 'val.csv')) 
        df_test = pd.read_csv(os.path.join(dataFolder, 'test.csv'))
    except Exception as e:
        raise ValueError(f"读取数据文件失败: {str(e)}")

    params = {'batch_size': args.batch_size, 'shuffle': True,
              'num_workers': args.workers, 'drop_last': True}
    training_generator = data.DataLoader(BIN_Data_Encoder(df_train.index, df_train.Label, df_train),**params)
    validation_generator = data.DataLoader(BIN_Data_Encoder(df_val.index, df_val.Label, df_val),**params)
    testing_generator = data.DataLoader(BIN_Data_Encoder(df_test.index, df_test.Label, df_test),**params)

    # 早停机制
    max_auc = 0
    best_threshold = 0.5
    model_max = copy.deepcopy(model)

    print('\n--- Starting Training ---')
    for epoch in range(args.epochs):
        # 训练循环
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0
        progress_bar = tqdm(
            enumerate(training_generator),
            total=len(training_generator),
            desc=f'Epoch {epoch + 1}',
            leave=True
        )
        for i, (d, p, d_mask, p_mask, label) in progress_bar:
            # 将数据移动到选定的设备
            d = d.long().to(device)
            p = p.long().to(device)
            d_mask = d_mask.long().to(device)
            p_mask = p_mask.long().to(device)
            label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
            
            opt.zero_grad()
            score = model(d, p, d_mask, p_mask)
            m = torch.nn.Sigmoid()
            logits = torch.squeeze(m(score))
            loss = torch.nn.BCELoss()(logits, label)
            loss.backward()
            opt.step()
            
            # 更新损失和批次计数
            epoch_train_loss += loss.item()
            batch_count += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'epoch': f'{epoch+1}/{args.epochs}',
                'avg_loss': f'{epoch_train_loss / batch_count:.4f}' if batch_count > 0 else '0.0000'
            })

            if i % 1000 == 0:
                print(f'Epoch {epoch + 1} Batch {i} Loss: {loss.item():.4f}')

        # 记录训练损失
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)

        # 验证阶段
        with torch.no_grad():
            auc_val, auprc_val, f1_val, _, val_loss, thred_optim, _ = test(validation_generator, model)
            val_aucs.append(auc_val)
            val_auprcs.append(auprc_val)
            val_f1s.append(f1_val)
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1} Validation | '
                  f'AUC: {auc_val:.4f} | AUPRC: {auprc_val:.4f} | '
                  f'F1: {f1_val:.4f} | Loss: {val_loss:.4f}')

            # 更新最佳模型和阈值
            if auc_val > max_auc:
                max_auc = auc_val
                best_threshold = thred_optim
                model_max = copy.deepcopy(model)

        break # =======================测试

    # 绘制训练曲线
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(val_aucs, color='orange')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')

    plt.subplot(2, 3, 3)
    plt.plot(val_auprcs, color='green')
    plt.title('Validation AUPRC')
    plt.xlabel('Epochs')
    plt.ylabel('AUPRC')

    plt.subplot(2, 3, 4)
    plt.plot(val_f1s, color='red')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

    # 最终测试并生成热力图
    print('\n--- Final Testing ---')
    with torch.no_grad():
        test_auc, test_auprc, test_f1, y_pred, test_loss, _, y_true = test(testing_generator, model_max,
                                                                           threshold=best_threshold)

        # 生成混淆矩阵热力图
        y_pred_labels = (np.array(y_pred) >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix_heatmap.png')
        plt.show()

        print(f'Test Results | AUC: {test_auc:.4f} | AUPRC: {test_auprc:.4f} | '
              f'F1: {test_f1:.4f} | Loss: {test_loss:.4f}')

    # 保存模型为h5格式
    print('\n--- Saving Model ---')
    
    # 创建保存目录
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(save_dir, f'model_{args.task}.h5')
    with h5py.File(model_path, 'w') as f:
        # 保存模型配置
        config_group = f.create_group('config')
        for key, value in config.items():
            config_group.attrs[key] = str(value)
        
        # 保存训练参数
        training_group = f.create_group('training')
        training_group.attrs['best_auc'] = max_auc
        training_group.attrs['best_threshold'] = best_threshold
        training_group.attrs['epochs'] = args.epochs
        
        # 保存模型权重
        weights_group = f.create_group('weights')
        for name, param in model_max.state_dict().items():
            weights_group.create_dataset(name, data=param.cpu().numpy())
    
    print(f'Model saved to {model_path}')

    return model_max, best_threshold


# In[5]:


# 删除在主模块外直接调用main()的代码
# start_time = time()
# best_model, best_threshold = main()
# print(f'Total time: {time() - start_time:.2f} seconds')


# In[6]:


def shap_explainer(model, test_generator, num_samples=10):
    """
    使用SHAP进行模型可解释性分析与可视化
    
    Args:
        model: 训练好的模型
        test_generator: 测试数据生成器
        num_samples: 用于分析的样本数量
    """
    # 导入SHAP库
    import shap
    import scipy.special as sp
    import time
    from tqdm.auto import tqdm
    
    # 设置绘图样式和字体大小
    plt.style.use('default')
    plt.rcParams.update({'font.size': 14})
    
    # 检测CUDA是否可用，如果可用则使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device for SHAP analysis: {device}")
    
    # 将模型移到选择的设备
    model = model.to(device)
    model.eval()
    
    print("准备SHAP分析数据...")
    start_time = time.time()
    
    # 收集用于分析的样本数据
    all_d = []
    all_p = []
    all_d_mask = []
    all_p_mask = []
    labels = []
    
    # 提取样本数据
    with torch.no_grad():
        for i, (d, p, d_mask, p_mask, label) in enumerate(test_generator):
            if i >= num_samples:
                break
                
            # 添加每个批次中的样本
            for j in range(len(d)):
                all_d.append(d[j].to(device))
                all_p.append(p[j].to(device))
                all_d_mask.append(d_mask[j].to(device))
                all_p_mask.append(p_mask[j].to(device))
                labels.append(label[j].item())
    
    # 获取序列长度
    drug_seq_len = len(all_d[0])
    protein_seq_len = len(all_p[0])
    
    # 计算总特征数
    num_features = drug_seq_len + protein_seq_len
    
    print(f"药物序列长度: {drug_seq_len}, 蛋白质序列长度: {protein_seq_len}")
    print(f"总特征数: {num_features}")
    print(f"收集到 {len(all_d)} 个样本用于分析")
    
    # 创建特征名称
    feature_names = []
    # 药物特征
    for i in range(drug_seq_len):
        feature_names.append(f"Drug_Pos_{i}")
    # 蛋白质特征
    for i in range(protein_seq_len):
        feature_names.append(f"Protein_Pos_{i}")
    
    print("计算特征重要性 - 使用GPU加速计算...")
    
    # 设置SHAP分析所需的参数
    num_analyze = min(5, len(all_d))
    
    # 创建真实背景数据
    print("准备背景数据...")
    background_size = min(len(all_d), 5)  # 使用较少的背景样本
    background_data = np.zeros((background_size, num_features))
    
    for i in range(background_size):
        # 合并药物和蛋白质数据为一个特征向量
        background_data[i, :drug_seq_len] = all_d[i].cpu().numpy()
        background_data[i, drug_seq_len:] = all_p[i].cpu().numpy()
    
    # 准备要解释的实例（选择不同于背景数据的样本）
    instances_data = np.zeros((num_analyze, num_features))
    for i in range(num_analyze):
        # 使用与背景数据不同的样本
        idx = min(i + background_size, len(all_d) - 1)
        instances_data[i, :drug_seq_len] = all_d[idx].cpu().numpy()
        instances_data[i, drug_seq_len:] = all_p[idx].cpu().numpy()
    
    # 计算原始预测值，用于结果展示
    predictions = []
    with torch.no_grad():
        for i in range(num_analyze):
            idx = min(i + background_size, len(all_d) - 1)
            output = model(all_d[idx].unsqueeze(0), all_p[idx].unsqueeze(0),
                          all_d_mask[idx].unsqueeze(0), all_p_mask[idx].unsqueeze(0))
            sigmoid_output = torch.sigmoid(output)
            if sigmoid_output.numel() == 1:
                predictions.append(sigmoid_output.item())
            else:
                predictions.append(sigmoid_output.mean().item())
    
    # 解决维度不匹配问题的关键 - 使用特定的模型包装器
    def model_wrapper(masker_x):
        """
        重要: 这个模型包装器函数必须确保返回与输入长度相同的结果
        masker_x: 输入数据，可能是背景数据或其扰动版本
        """
        with torch.no_grad():
            # 获取当前输入批次大小
            current_batch_size = masker_x.shape[0]
            
            # 确保结果数组与输入批次大小匹配
            results = np.zeros(current_batch_size)
            
            # 处理每个输入样本
            for i in range(current_batch_size):
                x_single = masker_x[i:i+1]  # 取单个样本，保持2D形状
                
                # 将输入从NumPy数组转换为PyTorch张量
                x_tensor = torch.tensor(x_single, dtype=torch.float32).to(device)
                
                # 分离药物和蛋白质数据
                input_d = x_tensor[:, :drug_seq_len].long()
                input_p = x_tensor[:, drug_seq_len:].long()
                
                # 创建掩码 (全1)
                input_d_mask = torch.ones((1, drug_seq_len), device=device).long()
                input_p_mask = torch.ones((1, protein_seq_len), device=device).long()
                
                # 运行模型
                output = model(input_d, input_p, input_d_mask, input_p_mask)
                
                # 应用sigmoid并确保是标量
                sigmoid_output = torch.sigmoid(output)
                
                # 提取标量值
                if sigmoid_output.numel() == 1:
                    results[i] = sigmoid_output.item()
                else:
                    results[i] = sigmoid_output.mean().item()

                break # ==================================测试
            return results
    
    # 使用KernelExplainer (最可靠且与任何模型兼容)
    print("使用KernelExplainer计算SHAP值...")
    
    # 先测试模型包装器，确保输出形状正确
    print("验证模型输出...")
    test_out = model_wrapper(background_data)
    print(f"背景数据形状: {background_data.shape}")
    print(f"模型输出形状: {test_out.shape}")
    
    # 确保输出与背景数据长度匹配
    if len(test_out) != len(background_data):
        raise ValueError(f"模型输出长度 {len(test_out)} 与背景数据长度 {len(background_data)} 不匹配!")
    
    try:
        # 创建KernelExplainer并使用验证过的模型包装器
        explainer = shap.KernelExplainer(
            model_wrapper, 
            background_data,
            link='identity'
        )
        
        # 使用较大的nsamples值提高精度，但不要过大以避免GPU内存问题
        print("计算样本SHAP值 (使用GPU加速中)...")
        shap_values = explainer.shap_values(instances_data, nsamples=200)
        
        # 确保SHAP值是numpy数组
        if not isinstance(shap_values, np.ndarray):
            shap_values = np.array(shap_values)
        
        # 确保SHAP值具有正确的维度 [样本数, 特征数]
        if len(shap_values.shape) > 2:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
    except Exception as e:
        print(f"SHAP计算出错: {str(e)}")
        print("尝试使用DeepExplainer替代方案...")
        
        # 创建特殊的背景和实例张量
        background_tensor_d = torch.stack([d for d in all_d[:background_size]])
        background_tensor_p = torch.stack([p for p in all_p[:background_size]])
        background_tensor_d_mask = torch.stack([m for m in all_d_mask[:background_size]])
        background_tensor_p_mask = torch.stack([m for m in all_p_mask[:background_size]])
        
        instance_tensor_d = torch.stack([all_d[min(i + background_size, len(all_d) - 1)] for i in range(num_analyze)])
        instance_tensor_p = torch.stack([all_p[min(i + background_size, len(all_p) - 1)] for i in range(num_analyze)])
        instance_tensor_d_mask = torch.stack([all_d_mask[min(i + background_size, len(all_d_mask) - 1)] for i in range(num_analyze)])
        instance_tensor_p_mask = torch.stack([all_p_mask[min(i + background_size, len(all_p_mask) - 1)] for i in range(num_analyze)])
        
        try:
            # 定义特殊包装模型函数
            def wrapped_model(d, p):
                return torch.sigmoid(model(d, p, 
                                          torch.ones((d.shape[0], drug_seq_len), device=device).long(),
                                          torch.ones((p.shape[0], protein_seq_len), device=device).long()))
            
            # 创建DeepExplainer
            explainer = shap.DeepExplainer(wrapped_model, 
                                          [background_tensor_d, background_tensor_p])
            
            # 计算SHAP值
            shap_values = explainer.shap_values([instance_tensor_d, instance_tensor_p])
            
            # 处理药物和蛋白质的SHAP值
            drug_shap = shap_values[0]
            protein_shap = shap_values[1]
            
            # 合并SHAP值为一个统一数组
            combined_shap = []
            for i in range(len(drug_shap)):
                combined_shap.append(np.concatenate([drug_shap[i].flatten(), protein_shap[i].flatten()]))
            
            shap_values = np.array(combined_shap)
        
        except Exception as e:
            print(f"DeepExplainer也失败: {str(e)}")
            print("回退到手动计算特征重要性...")
            
            # 创建简单的特征重要性矩阵
            shap_values = np.zeros((num_analyze, num_features))
            
            # 对于每个要分析的样本
            for i in range(num_analyze):
                # 获取基准预测
                idx = min(i + background_size, len(all_d) - 1)
                baseline_pred = predictions[i]
                
                # 对于每个特征，计算简单的扰动重要性
                for j in range(num_features):
                    # 创建扰动副本
                    perturbed_instance = instances_data[i].copy()
                    
                    # 扰动特征 (设置为0或均值)
                    if j < drug_seq_len:
                        # 药物特征
                        orig_val = perturbed_instance[j]
                        perturbed_instance[j] = 0  # 或使用背景数据的均值
                    else:
                        # 蛋白质特征
                        orig_val = perturbed_instance[j]
                        perturbed_instance[j] = 0  # 或使用背景数据的均值
                    
                    # 计算扰动后的预测
                    perturbed_pred = model_wrapper(perturbed_instance.reshape(1, -1))[0]
                    
                    # 计算特征重要性 (预测差异)
                    shap_values[i, j] = baseline_pred - perturbed_pred
            
            # 创建简单的Explainer对象，只包含expected_value
            class SimpleExplainer:
                def __init__(self, expected_value):
                    self.expected_value = expected_value
            
            # 使用平均预测作为基准值
            explainer = SimpleExplainer(np.mean(predictions))
    
    # 计算时间
    elapsed_time = time.time() - start_time
    print(f"SHAP值计算完成，用时: {elapsed_time:.2f} 秒")
    
    # 如果SHAP值很小，放大它们以便更好地可视化
    max_abs_shap = np.max(np.abs(shap_values))
    if max_abs_shap < 0.01:
        print(f"SHAP值较小 (最大值: {max_abs_shap:.6f})，放大以便可视化...")
        scaling_factor = 100.0
        shap_values = shap_values * scaling_factor
        print(f"应用缩放因子 {scaling_factor}")
    
    # 1. Force Plot - 特征影响力
    print("绘制Force Plot...")
    
    import IPython
    # 使用原生SHAP库的force_plot
    # 选择一个样本进行可视化
    sample_idx = 0
    
    # 准备基准值（expected value）
    base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[0]
    
    # 使用SHAP库的原生force_plot
    force_plot = shap.force_plot(
        base_value,
        shap_values[sample_idx, :],
        instances_data[sample_idx, :],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        figsize=(20, 8)
    )
    plt.tight_layout()
    plt.savefig('shap_force_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Summary Plot - 特征重要性概览
    print("绘制Summary Plot...")
    
    # 使用SHAP库的原生summary_plot，但移除random_state参数
    plt.figure(figsize=(14, 10))
    try:
        # 首先尝试使用较新版本的参数
        shap.summary_plot(
            shap_values,
            instances_data,
            feature_names=feature_names,
            max_display=30,
            show=False,
            random_state=np.random.RandomState(42)
        )
    except TypeError:
        # 如果失败，回退到没有random_state参数的调用
        print("警告: SHAP库版本可能较旧，使用兼容模式...")
        shap.summary_plot(
            shap_values,
            instances_data,
            feature_names=feature_names,
            max_display=30,
            show=False
        )
    
    # 调整布局参数，增加左侧边距
    plt.subplots_adjust(left=0.3)
    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Dependency Plot - 特征依赖关系
    print("绘制Dependency Plot...")
    
    if len(feature_names) >= 2 and shap_values.shape[0] > 1:
        # 计算平均特征重要性
        mean_imp = np.mean(np.abs(shap_values), axis=0)
        # 获取两个最重要特征的索引
        sorted_idx = np.argsort(mean_imp)[::-1]
        feature1_idx = sorted_idx[0]
        feature2_idx = sorted_idx[1]
        
        # 使用SHAP库的原生dependence_plot
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            feature1_idx,
            shap_values,
            instances_data,
            feature_names=feature_names,
            interaction_index=feature2_idx,
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_dependency_plot.png', dpi=300)
        plt.show()
    
    # 4. Waterfall Plot - 预测贡献瀑布图
    print("绘制Waterfall Plot...")
    
    # 使用SHAP库的原生waterfall_plot (仅限于shap 0.40.0+版本)
    try:
        sample_idx = 0
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[sample_idx],
                base_values=base_value,
                data=instances_data[sample_idx],
                feature_names=feature_names
            ),
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_waterfall_plot.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"无法使用原生waterfall_plot: {e}") 
        print("尝试使用自定义瀑布图...")
        
        # 选择前15个特征
        sample_idx = 0
        waterfall_feature_count = min(15, shap_values.shape[1])
        sorted_idx = np.argsort(np.abs(shap_values[sample_idx]))[::-1][:waterfall_feature_count]
        sorted_values = shap_values[sample_idx][sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        plt.figure(figsize=(12, 8))
        
        # 条形图
        colors = ['red' if x > 0 else 'blue' for x in sorted_values]
        plt.barh(range(waterfall_feature_count), sorted_values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-')
        plt.yticks(range(waterfall_feature_count), sorted_names)
        plt.xlabel('Feature Impact (+ Increase Prediction, - Decrease Prediction)')
        
        # 使用正确的标签索引
        idx = min(sample_idx + background_size, len(labels) - 1)
        plt.title(f'SHAP Waterfall Plot - Sample Feature Contribution\n(Label: {labels[idx]}, Prediction: {predictions[sample_idx]:.4f})')
        
        plt.subplots_adjust(left=0.3)
        plt.savefig('shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Bar Plot - 特征重要性排序
    print("绘制Bar Plot...")
    
    # 使用SHAP库的原生bar_plot (仅限于shap 0.40.0+版本)
    try:
        plt.figure(figsize=(12, 8))
        # 确保shap_values不是单个浮点数
        if isinstance(shap_values, (float, int)):
            print("SHAP值是一个标量，无法使用bar_plot")
            raise ValueError("SHAP值格式不正确")
        
        # 创建正确格式的Explanation对象
        exp = shap.Explanation(
            values=shap_values,
            base_values=np.array([base_value] * len(shap_values)),  # 确保base_values是数组
            data=instances_data,
            feature_names=feature_names
        )
        shap.plots.bar(exp, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=300)
        plt.show()
    except Exception as e:
        print(f"无法使用原生bar_plot: {e}")
        # 使用自定义图表(保留原有的自定义代码)
        
        # 计算平均特征重要性
        mean_imp = np.mean(shap_values, axis=0)
        # 按绝对重要性排序
        abs_imp = np.abs(mean_imp)
        sorted_idx = np.argsort(abs_imp)[::-1]
        top_10_idx = sorted_idx[:10]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_10_idx)), abs_imp[top_10_idx], color='blue')
        plt.yticks(range(len(top_10_idx)), [feature_names[i] for i in top_10_idx])
        plt.xlabel('Absolute Feature Importance')
        plt.title('Feature Importance Ranking - Top 10 Features')
        
        # 添加值标签
        for i, value in enumerate(abs_imp[top_10_idx]):
            plt.text(value + 0.001, i, f'{value:.3f}', va='center')
        
        plt.subplots_adjust(left=0.3)
        plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 6. 全部特征影响力分布图 - 使用SHAP库的原生beeswarm_plot
    print("绘制特征影响力分布图...")
    
    plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values,
            base_values=np.full(shap_values.shape[0], base_value),
            data=instances_data,
            feature_names=feature_names
        ),
        max_display=20,
        show=False
    )
    plt.tight_layout()
    plt.savefig('shap_beeswarm_plot.png', dpi=300)
    plt.show()
    
    # 统计信息
    print("\n=== SHAP分析统计 ===")
    print(f"分析的特征总数: {shap_values.shape[1]}")
    print(f"- 药物序列位置: {drug_seq_len}")
    print(f"- 蛋白质序列位置: {protein_seq_len}")
    print(f"使用的样本数量: {len(shap_values)}")
    
    # 计算平均特征重要性
    mean_imp = np.mean(shap_values, axis=0)
    
    # 分开药物和蛋白质特征
    drug_features = mean_imp[:drug_seq_len]
    protein_features = mean_imp[drug_seq_len:]
    
    # 找出贡献最大的位置
    top_drug_idx = np.argmax(np.abs(drug_features))
    top_protein_idx = np.argmax(np.abs(protein_features))
    print(f"贡献最大的药物位置: Drug_Pos_{top_drug_idx} (重要性: {drug_features[top_drug_idx]:.5f})")
    print(f"贡献最大的蛋白质位置: Protein_Pos_{top_protein_idx} (重要性: {protein_features[top_protein_idx]:.5f})")
    
    # 分布统计
    print("\n--- 值分布 ---")
    positive_features = sum(1 for x in mean_imp if x > 0)
    negative_features = sum(1 for x in mean_imp if x < 0)
    print(f"正向影响特征: {positive_features} ({positive_features/len(mean_imp)*100:.1f}%)")
    print(f"负向影响特征: {negative_features} ({negative_features/len(mean_imp)*100:.1f}%)")
    print(f"平均绝对重要性: {np.mean(np.abs(mean_imp)):.5f}")
    print(f"最大绝对重要性: {np.max(np.abs(mean_imp)):.5f}")
    print(f"总计算时间: {elapsed_time:.2f} 秒")
    print("===============================")
    
    print("\nSHAP分析完成! 所有可视化图表已保存:")
    print("1. shap_force_plot.png - 特征影响力")
    print("2. shap_summary_plot.png - 特征重要性概览")
    print("3. shap_dependency_plot.png - 特征依赖关系")
    print("4. shap_waterfall_plot.png - 样本特征贡献")
    print("5. shap_bar_plot.png - 特征重要性排序")
    print("6. shap_beeswarm_plot.png - 特征影响力分布")
    
    return explainer, shap_values


# In[7]:


# Run SHAP analysis after training
# Use a small number of samples for demonstration purposes

# Make sure necessary classes and functions are imported
# from stream import BIN_Data_Encoder
# from config import BIN_config_DBPE

# # Reload test data, as df_test is defined inside the main() function
# dataFolder = get_task(args.task)  # Get data folder path
# df_test = pd.read_csv(dataFolder + '/test.csv')  # Reload test data

# # Use very small batch size to avoid memory issues
# small_batch_size = 2  # Use smaller batch size

# testing_generator = data.DataLoader(
#     BIN_Data_Encoder(df_test.index, df_test.Label, df_test),
#     batch_size=small_batch_size, 
#     shuffle=False,
#     num_workers=1,  # Reduce worker threads
#     drop_last=False
# )

# print("Starting SHAP interpretability analysis...")
# print("Note: Processing all features may take some time, please be patient...")
# # Use fewer samples to speed up analysis
# explainer, shap_values = shap_explainer(best_model, testing_generator, num_samples=30)
# print("SHAP analysis completed!")


# In[ ]:


# 修改文件末尾，添加正确的主模块保护
if __name__ == '__main__':
    import os
    import h5py
    import torch
    from config import BIN_config_DBPE
    from models import BIN_Interaction_Flat
    import os
    import h5py 
    import multiprocessing
    # 添加多进程支持
    multiprocessing.freeze_support()
    # 设置多进程启动方法为spawn（在Windows上特别重要）
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        # 如果已经设置过启动方法，会抛出RuntimeError
        pass
    
    # 减少worker数量，避免内存过载
    args.workers = min(4, args.workers)
    
    # 调用主函数
    start_time = time()
    best_model, best_threshold = main()
    print(f'Total time: {time() - start_time:.2f} seconds')
    
    # 保存模型（如果需要的话）
    if best_model is not None:
        print("训练完成，保存最佳模型...")
        os.makedirs('saved_models', exist_ok=True)
        model_save_path = f"saved_models/model_{args.task}.h5"
        with h5py.File(model_save_path, 'w') as f:
            g = f.create_group('weights')
            for name, param in best_model.state_dict().items():
                g.create_dataset(name, data=param.cpu().numpy())
            
            # 创建训练信息组
            train_group = f.create_group('training')
            # 保存阈值信息
            train_group.attrs['best_threshold'] = best_threshold
        
        print(f"模型已保存到: {model_save_path}")
    else:
        print("训练未生成有效模型。")

#  # 加载配置
#     config = BIN_config_DBPE()
#     config['batch_size'] = args.batch_size
    
#     # 初始化模型
#     model = BIN_Interaction_Flat(**config).to(device)
#     # 从保存的h5文件加载模型权重
#     model_path = f"saved_models/model_{args.task}.h5"
#     if os.path.exists(model_path):
#         print(f"正在从 {model_path} 加载模型...")
#         with h5py.File(model_path, 'r') as f:
#             # 加载权重
#             weights_group = f['weights']
#             state_dict = {}
#             for name in weights_group:
#                 state_dict[name] = torch.tensor(weights_group[name][()])
            
#             # 加载模型状态
#             model.load_state_dict(state_dict)
            
#             # 如果存在阈值信息，也加载
#             best_threshold = 0.5
#             if 'training' in f and 'best_threshold' in f['training'].attrs:
#                 best_threshold = f['training'].attrs['best_threshold']
#                 print(f"使用保存的最佳阈值: {best_threshold}")
#             print("模型加载成功!")
#     else:
#         print(f"错误: 找不到模型文件 {model_path}")
#         print("请先训练并保存模型，或确保文件路径正确。")
#         exit(1)
#     model.eval()  # 设置为评估模式

#     from stream import BIN_Data_Encoder
#     from config import BIN_config_DBPE

#     # Reload test data, as df_test is defined inside the main() function
#     dataFolder = get_task(args.task)  # Get data folder path
#     df_test = pd.read_csv(dataFolder + '/test.csv')  # Reload test data

#     # Use very small batch size to avoid memory issues
#     small_batch_size = 2  # Use smaller batch size

#     testing_generator = data.DataLoader(
#         BIN_Data_Encoder(df_test.index, df_test.Label, df_test),
#         batch_size=small_batch_size, 
#         shuffle=False,
#         num_workers=1,  # Reduce worker threads
#         drop_last=False
#     )

#     print("Starting SHAP interpretability analysis...")
#     print("Note: Processing all features may take some time, please be patient...")
#     # Use fewer samples to speed up analysis
#     # explainer, shap_values = shap_explainer(best_model, testing_generator, num_samples=30)
#     explainer, shap_values = shap_explainer(model, testing_generator, num_samples=30)
#     print("SHAP analysis completed!")


# In[ ]:





# In[ ]:




