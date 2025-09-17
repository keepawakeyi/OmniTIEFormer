# -*- coding: utf-8 -*-
"""
FEDformer模型的超参数调优代码
- 使用Excel和JSON格式存储结果，便于阅读
"""
import os
import json
import itertools
from datetime import datetime
from assistant import get_gpus_memory_info

id, _ = get_gpus_memory_info()
os.environ["CUDA_VISIBLE_DEVICES"] = str(id)

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import EncoderNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE
import warnings
warnings.filterwarnings("ignore")

from DataPreProcessPanasonic import BatteryDataProcess
from ModelsModify.FEDformer import FEDFormerNetModel
from assistant import set_seed

import argparse

# FEDformer超参数搜索空间定义
# 只包含FEDFormerNetModel中定义的参数
HYPERPARAMETER_SPACE = {
    'seq_len': [32, 64, 128],                     # 输入序列长度
    'label_len': [0, 16, 32],                     # 标签长度
    'pred_len': [1, 3, 5],                        # 预测长度
    'd_model': [16, 32, 64],                      # 模型维度
    'factor': [1, 3, 5],                          # attention因子
    'dropout': [0.01, 0.05, 0.1],                 # dropout率
    'd_ff': [32, 64, 128],                        # 前馈网络维度
    'e_layers': [1, 2, 3],                        # 编码器层数
    'd_layers': [1, 2],                           # 解码器层数
    'top_k': [3, 5, 10],                          # FEDformer特有参数
    'num_kernels': [4, 6, 8],                     # FEDformer特有参数
    'batch_size': [64, 128, 256],                 # 批次大小
    'learning_rate': [0.001, 0.0005, 0.0001],     # 学习率
}

class FEDformerOptimizer:
    """FEDformer RUL预测优化器，用于系统化的超参数调优和重复实验"""
    
    def __init__(self, args):
        self.args = args
        self.results_dir = f'hyperparameter_tuning_results/{args.model}_{args.test_name}'
        self.create_directories()
        
    def create_directories(self):
        """创建必要的目录结构"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f'{self.results_dir}/logs', exist_ok=True)
        os.makedirs(f'{self.results_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.results_dir}/predictions', exist_ok=True)
        os.makedirs(f'{self.results_dir}/final_results', exist_ok=True)
        
    def rul_value_error(self, y_test, y_predict, threshold):
        """计算RUL相关误差"""
        true_re, pred_re = len(y_test), 0
        
        for i in range(len(y_test)-1):
            if y_test[i] <= threshold >= y_test[i+1]:
                true_re = i - 1
                break
                
        for i in range(len(y_predict)-1):
            if y_predict[i] <= threshold:
                pred_re = i - 1
                break
                
        rul_real = true_re + 1
        rul_pred = pred_re + 1
        ae_error = abs(true_re - pred_re)        
        re_score = abs(true_re - pred_re) / true_re if true_re > 0 else 1
        if re_score > 1: 
            re_score = 1
            
        return rul_real, rul_pred, ae_error, re_score
    
    def prepare_data(self, battery_data, start_point, hyperparams):
        """准备数据集"""
        df_train, df_test, df_all = BatteryDataProcess(
            battery_data, self.args.test_name, start_point
        )
        
        mask_len = len(df_train)
        time_varying_known_reals = ['Capacity']
        time_varying_unknown_reals = ['target']
        
        # 使用超参数中的值
        max_prediction_length = hyperparams['pred_len']
        max_encoder_length = hyperparams['seq_len']
        
        # 创建训练集
        training = TimeSeriesDataSet(
            df_train[0:int(0.8*mask_len)],
            time_idx="time_idx",
            target="target",
            group_ids=['group_id'],  
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length, 
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=EncoderNormalizer(),
            add_encoder_length=False
        )
        
        # 创建验证集
        validing = TimeSeriesDataSet(
            df_train[int(0.8 * mask_len):],
            time_idx="time_idx",
            target='target', 
            group_ids=['group_id'],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length, 
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=EncoderNormalizer(),
            add_encoder_length=False
        )
        
        # 创建测试集
        testing = TimeSeriesDataSet(
            df_test,
            time_idx="time_idx",
            target='target', 
            group_ids=['group_id'],
            min_encoder_length=max_encoder_length,
            max_encoder_length=max_encoder_length,
            min_prediction_length=max_prediction_length,  
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=EncoderNormalizer(),
            add_encoder_length=False
        )
        
        return training, validing, testing, df_all
    
    def train_single_model(self, training, validing, testing, df_all, start_point, hyperparams, run_seed=None):
        """训练单个模型并返回结果"""
        # 设置随机种子
        if run_seed is not None:
            set_seed(run_seed)
        
        # 创建数据加载器
        train_dataloader = training.to_dataloader(
            train=True, 
            batch_size=hyperparams['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        val_dataloader = validing.to_dataloader(
            train=False,
            batch_size=hyperparams['batch_size'],
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        test_dataloader = testing.to_dataloader(
            train=False,
            batch_size=hyperparams['batch_size'],
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        # 创建FEDformer模型
        # 设置编码器和解码器的输入维度
        enc_in = len(training.time_varying_known_reals)
        dec_in = enc_in
        c_out = 1  # 单变量预测
        
        model = FEDFormerNetModel.from_dataset(
            training,
            seq_len=hyperparams['seq_len'],
            label_len=hyperparams['label_len'],
            pred_len=hyperparams['pred_len'],
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            e_layers=hyperparams['e_layers'],
            d_layers=hyperparams['d_layers'],
            factor=hyperparams['factor'],
            d_model=hyperparams['d_model'],
            d_ff=hyperparams['d_ff'],
            dropout=hyperparams['dropout'],
            top_k=hyperparams['top_k'],
            num_kernels=hyperparams['num_kernels'],
            learning_rate=hyperparams['learning_rate'],
            loss=SMAPE(),
        )
        
        # 设置回调
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-5,
            patience=10,
            verbose=False,
            mode='min'
        )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.args.max_epochs,
            gpus=1,
            gradient_clip_val=0.2,
            callbacks=[early_stop_callback],
            logger=False,
            enable_checkpointing=True,
            default_root_dir=f"{self.results_dir}/checkpoints"
        )
        
        # 训练模型
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        
        # 加载最佳模型
        best_model_path = trainer.checkpoint_callback.best_model_path
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_model = FEDFormerNetModel.load_from_checkpoint(best_model_path).to(device=device)
        
        # 预测
        predictions = best_model.predict(test_dataloader, batch_size=256)
        predictions = predictions.detach().cpu().numpy().reshape(-1)
        
        # 获取实际值
        actuals_df = df_all.loc[df_all['Cycle'] >= start_point, ['Cycle', 'target']]
        actuals = actuals_df['target'].values
        y_true = actuals * self.args.Rated_Capacity
        y_pred = predictions * self.args.Rated_Capacity
        
        # 计算指标
        mask = y_true >= 0.
        results = self.calculate_metrics(y_true[mask], y_pred[mask])
        results['epochs_trained'] = trainer.current_epoch
        results['best_model_path'] = best_model_path
        
        return results, y_pred
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        from sklearn.metrics import r2_score
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
        r2 = r2_score(y_true, y_pred)
        
        rul_real, rul_pred, ae, re = self.rul_value_error(
            y_true, y_pred, 
            threshold=self.args.Rated_Capacity * 0.7
        )
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'RUL_real': rul_real,
            'RUL_pred': rul_pred,
            'AE': ae,
            'RE': re
        }
    
    def run_hyperparameter_search(self, battery_data, param_grid):
        """运行超参数搜索（只在起始点300，每个组合只做1次）"""
        print("=== 开始FEDformer超参数搜索 ===")
        print(f"搜索起始点: 300")
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        all_results = []
        start_point = 300  # 固定使用起始点300
        
        print(f"总共需要测试 {len(param_combinations)} 个参数组合")
        
        for i, param_combo in enumerate(param_combinations):
            hyperparams = dict(zip(param_names, param_combo))
            
            print(f"\n超参数实验 {i+1}/{len(param_combinations)}")
            print(f"参数: {hyperparams}")
            
            # 准备数据
            training, validing, testing, df_all = self.prepare_data(
                battery_data, start_point, hyperparams
            )
            
            # 运行单次实验
            set_seed(1)  # 固定种子确保可重复性
            
            try:
                results, y_pred = self.train_single_model(
                    training, validing, testing, df_all, 
                    start_point, hyperparams
                )
                
                if results is not None:
                    result_entry = {
                        'hyperparameters': hyperparams,
                        'start_point': start_point,
                        'results': results,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    all_results.append(result_entry)
                    
                    # 保存单次预测结果为CSV
                    param_str = f"sl{hyperparams['seq_len']}_pl{hyperparams['pred_len']}_dm{hyperparams['d_model']}_el{hyperparams['e_layers']}_dl{hyperparams['d_layers']}_tk{hyperparams['top_k']}"
                    filename = f'{self.results_dir}/predictions/hyperparam_search_SP{start_point}_{i+1}_{param_str}.csv'
                    pred_df = pd.DataFrame({
                        'cycle': range(len(y_pred)),
                        'prediction': y_pred,
                        'hyperparams': str(hyperparams)
                    })
                    pred_df.to_csv(filename, index=False)
                    
                    # 保存中间结果
                    self.save_hyperparameter_results(all_results)
                    
                    print(f"  MAE: {results['MAE']:.4f}, RMSE: {results['RMSE']:.4f}, R2: {results['R2']:.4f}")
                    
            except Exception as e:
                print(f"实验失败: {e}")
                continue
        
        return all_results
    
    def run_best_hyperparameter_experiments(self, battery_data, best_hyperparams):
        """使用最佳超参数在多个起始点运行重复实验"""
        print("\n=== 使用最佳超参数进行重复实验 ===")
        print(f"最佳超参数: {best_hyperparams}")
        
        start_points = [300, 400, 500]
        num_runs = 5
        
        # 存储所有结果
        all_results = {}
        final_predictions = {}
        
        for start_point in start_points:
            print(f"\n起始点: {start_point}")
            
            # 准备数据
            training, validing, testing, df_all = self.prepare_data(
                battery_data, start_point, best_hyperparams
            )
            
            # 运行多次实验
            sp_results = []
            sp_predictions = []
            
            for run in range(num_runs):
                print(f"  第 {run+1}/{num_runs} 次实验")
                
                try:
                    results, y_pred = self.train_single_model(
                        training, validing, testing, df_all, 
                        start_point, best_hyperparams, run_seed=run+1
                    )
                    
                    if results is not None:
                        sp_results.append(results)
                        sp_predictions.append(y_pred)
                        print(f"    MAE: {results['MAE']:.4f}, RMSE: {results['RMSE']:.4f}")
                        
                except Exception as e:
                    print(f"  实验失败: {e}")
                    continue
            
            # 计算平均结果
            if sp_results:
                avg_results = self.average_results(sp_results)
                all_results[f'SP{start_point}'] = {
                    'average_metrics': avg_results,
                    'individual_results': sp_results,
                    'num_runs': len(sp_results)
                }
                
                # 保存预测结果
                final_predictions[f'SP{start_point}'] = sp_predictions
                
                # 打印平均结果
                print(f"\n起始点 {start_point} 的平均结果:")
                for metric, value in avg_results.items():
                    if not metric.endswith('_std'):
                        print(f"  {metric}: {value:.4f}")
        
        return all_results, final_predictions
    
    def average_results(self, results_list):
        """计算多次实验的平均结果"""
        avg_results = {}
        metric_names = ['MAE', 'RMSE', 'R2', 'RUL_real', 'RUL_pred', 'AE', 'RE', 'epochs_trained']
        
        for metric in metric_names:
            if metric in results_list[0]:
                values = [r[metric] for r in results_list]
                avg_results[metric] = np.mean(values)
                avg_results[f'{metric}_std'] = np.std(values)
        
        return avg_results
    
    def save_hyperparameter_results(self, results):
        """保存超参数搜索结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.results_dir}/hyperparameter_results_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        # 同时保存一份最新的结果
        with open(f'{self.results_dir}/latest_hyperparameter_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    
    def save_final_results(self, results, predictions):
        """保存最终的重复实验结果（使用易读格式）"""
        # 保存结果到JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'{self.results_dir}/final_results/final_results_{timestamp}.json'
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        # 创建Excel文件保存所有预测值
        excel_filename = f'{self.results_dir}/final_results/RUL_{self.args.test_name}_{self.args.model}_predictions.xlsx'
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            for sp, pred_list in predictions.items():
                # 创建DataFrame
                pred_df = pd.DataFrame()
                for i, pred in enumerate(pred_list):
                    pred_df[f'Run_{i+1}'] = pred
                
                # 添加平均值和标准差
                pred_df['Mean'] = pred_df.mean(axis=1)
                pred_df['Std'] = pred_df.std(axis=1)
                
                # 写入Excel的不同sheet
                pred_df.to_excel(writer, sheet_name=sp, index_label='Cycle')
        
        # 也保存为JSON格式（更易于程序读取）
        predictions_json = {}
        for sp, pred_list in predictions.items():
            predictions_json[sp] = [pred.tolist() for pred in pred_list]
        
        json_pred_file = f'{self.results_dir}/final_results/RUL_{self.args.test_name}_{self.args.model}_predictions.json'
        with open(json_pred_file, 'w', encoding='utf-8') as f:
            json.dump(predictions_json, f, indent=2)
        
        # 创建汇总Excel文件
        summary_excel = f'{self.results_dir}/final_results/results_summary.xlsx'
        with pd.ExcelWriter(summary_excel, engine='openpyxl') as writer:
            # 汇总平均结果
            summary_data = []
            for sp, res in results.items():
                avg_metrics = res['average_metrics']
                row = {'StartPoint': sp}
                row.update(avg_metrics)
                summary_data.append(row)
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Average_Results', index=False)
            
            # 详细结果
            for sp, res in results.items():
                individual_df = pd.DataFrame(res['individual_results'])
                individual_df.to_excel(writer, sheet_name=f'{sp}_Details', index=False)
        
        print(f"\n结果已保存到: {self.results_dir}/final_results/")
        print(f"- Excel预测值: {excel_filename}")
        print(f"- JSON预测值: {json_pred_file}")
        print(f"- 结果汇总: {summary_excel}")
    
    def find_best_hyperparameters(self, results_file=None):
        """从超参数搜索结果中找出最佳MAE的参数"""
        if results_file is None:
            results_file = f'{self.results_dir}/latest_hyperparameter_results.json'
            
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # 按MAE排序找出最佳参数
        sorted_results = sorted(results, key=lambda x: x['results']['MAE'])
        best_result = sorted_results[0]
        
        best_hyperparams = best_result['hyperparameters'].copy()
        
        print(f"\n最佳MAE: {best_result['results']['MAE']:.4f}")
        print(f"最佳超参数: {best_hyperparams}")
        
        # 打印前5个最佳结果
        print("\n前5个最佳结果:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. MAE: {result['results']['MAE']:.4f}, 参数: {result['hyperparameters']}")
        
        return best_hyperparams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='FEDformer', help='Model name.')
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--root_dir', type=str, default='Panasonic_RUL_prediction_tuning', help='root path')
    parser.add_argument('--count', type=int, default=5, help='Number of independent experiments.')
    parser.add_argument('--data_dir', type=str, default='datasets/Panasonic/', help='path of the data')
    parser.add_argument('--Battery_list', type=list, default=['Cell01', 'Cell02', 'Cell03'])
    parser.add_argument('--Rated_Capacity', type=float, default=3, help='Rate Capacity')
    parser.add_argument('--test_name', type=str, default='Cell01', help='Battery for test')
    parser.add_argument('--max_epochs', type=int, default=100, help='max train epochs')
    
    args = parser.parse_args()
    
    # 加载电池数据
    battery_data = np.load('E:\DATA-CODE\PatchFormer-main\datasets\Panasonic\Panasonic_Data.npy', 
                          allow_pickle=True).item()
    
    # 创建优化器
    optimizer = FEDformerOptimizer(args)
    
    # 定义自定义参数搜索空间 - 针对FEDformer的特定参数
    custom_param_grid = {
        'seq_len': [64],                          # 输入序列长度
        'label_len': [0],                     # 标签长度
        'pred_len': [1],                          # 预测长度
        'd_model': [32, 16,8],                      # 模型维度
        'factor': [3, 5],                         # attention因子
        'dropout': [0.1],                   # dropout率
        'd_ff': [64, 128],                        # 前馈网络维度
        'e_layers': [1, 2],                       # 编码器层数
        'd_layers': [1],                          # 解码器层数
        'top_k': [5],                             # FEDformer特有参数
        'num_kernels': [6],                       # FEDformer特有参数
        'batch_size': [128],                      # 批次大小
        'learning_rate': [0.001]          # 学习率
    }
    
    # 步骤1：运行超参数搜索（只在起始点300，每个组合1次实验）
    print("步骤1：FEDformer超参数搜索")
    hyperparam_results = optimizer.run_hyperparameter_search(battery_data, custom_param_grid)
    
    # 步骤2：找出最佳MAE的超参数
    print("\n步骤2：选择最佳超参数")
    best_hyperparams = optimizer.find_best_hyperparameters()
    
    # 步骤3：使用最佳超参数在多个起始点进行重复实验
    print("\n步骤3：使用最佳超参数进行重复实验")
    final_results, final_predictions = optimizer.run_best_hyperparameter_experiments(
        battery_data, best_hyperparams
    )
    
    # 步骤4：保存最终结果
    optimizer.save_final_results(final_results, final_predictions)
    
    # 打印总结
    print("\n=== FEDformer实验完成 ===")
    print(f"所有结果保存在: {optimizer.results_dir}")
    print("\n各起始点的平均结果:")
    for sp, res in final_results.items():
        avg_metrics = res['average_metrics']
        print(f"\n{sp}:")
        print(f"  MAE: {avg_metrics['MAE']:.4f} ± {avg_metrics['MAE_std']:.4f}")
        print(f"  RMSE: {avg_metrics['RMSE']:.4f} ± {avg_metrics['RMSE_std']:.4f}")
        print(f"  R2: {avg_metrics['R2']:.4f} ± {avg_metrics['R2_std']:.4f}")
        print(f"  RE: {avg_metrics['RE']:.4f} ± {avg_metrics['RE_std']:.4f}")


if __name__ == "__main__":
    main()
