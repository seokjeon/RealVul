from argparse import ArgumentParser
import os,sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from pytorch_lightning import seed_everything, Trainer
import torch
torch.set_float32_matmul_precision('medium')
from src.models.vd import DeepWuKong
from src.datas.datamodules import XFGDataModule
from src.utils import filter_warnings
import os
import csv


def test(checkpoint_path: str, root_folder_path: str = None,split_folder_name: str = None, batch_size: int = None,save_predictions: bool = False, output_file: str = "deepwukong_predictions.txt"):
    """

    test the trained model using specified files

    Args:
        checkpoint_path:
        data_folder:
        batch_size:

    Returns:

    """
    filter_warnings()
    model = DeepWuKong.load_from_checkpoint(checkpoint_path)
    config = model.hparams["config"]
    vocabulary = model.hparams["vocab"]
    print(f"Setting batch size to: {batch_size}")  # 디버깅
    if batch_size is not None:
        config.hyper_parameters.test_batch_size = batch_size
    print(f"After setting: {config.hyper_parameters.test_batch_size}")  # 확인
    if root_folder_path is not None:
        config.root_folder_path = root_folder_path
    if split_folder_name is not None:
        config.split_folder_name = split_folder_name
    

   
    data_module = XFGDataModule(config, vocabulary)
    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(accelerator="gpu", devices=1, precision="16")
    
    if save_predictions:
        print("start prediction (saving individual predictions)")
        predictions = trainer.predict(model, datamodule=data_module)
        processed_predictions = {
            # unique_id.c: { 'file_paths': [], preds: [], pred: 0 or 1}
        }

        def _to_py(value):
            if torch.is_tensor(value):
                if value.numel() == 1:
                    return value.item()
                return value.tolist()
            return value

        for batch in predictions:
            preds = list(batch['preds'])
            file_paths = batch['file_paths']
            labels = batch['labels']
            for file_path, pred, label in zip(file_paths, preds, labels):
                parts = file_path.split('/')
                for part in parts:
                    if part.endswith('.c'):
                        unique_id = part[:-2]  # remove .c
                        break
                else:
                    continue  # .c가 없으면 skip

                if unique_id not in processed_predictions:
                    processed_predictions[unique_id] = {'file_paths': [], 'preds': [], 'labels': []}
                processed_predictions[unique_id]['file_paths'].append(file_path)
                processed_predictions[unique_id]['preds'].append(_to_py(pred))
                processed_predictions[unique_id]['labels'].append(_to_py(label))

        with open('deepwukong_individual_predictions.json', 'w') as pred_file:
            import json
            json.dump(processed_predictions, pred_file, indent=4)
        print("Individual predictions saved to deepwukong_individual_predictions.json")

        # # 최종 pred 계산 (하나라도 1이면 1)
        # for unique_id, v in processed_predictions.items():
        #     v['pred'] = 1 if any(p == 1 for p in v['preds']) else 0
            

        # csv_path = './downloads/RealVul/datasets/VP-Bench_Test_Dataset/Real_Vul_data.csv'
        # df = pd.read_csv(csv_path)

        # df['unique_id'] = df['unique_id'].astype(str)

        # df['deepwukong_files'] = df['unique_id'].map(
        #     lambda uid: ';'.join(processed_predictions.get(uid, {}).get('file_paths', []))
        # )
        # df['deepwukong_preds'] = df['unique_id'].map(
        #     lambda uid: ';'.join(map(str, processed_predictions.get(uid, {}).get('preds', [])))
        # )
        # df['deepwukong_final_pred'] = df['unique_id'].map(
        #     lambda uid: 'vulnerable' if processed_predictions.get(uid, {}).get('pred', 0) == 1 else 'normal'
        # )

        # # 새로운 파일로 저장
        # df.to_csv('Real_Vul_data_with_deepwukong.csv', index=False)
        # print("Merged CSV saved to Real_Vul_data_with_deepwukong.csv")

        # # Save predictions to file
        # with open(output_file, 'w') as f:
        #     for batch_idx, batch_preds in enumerate(predictions):
        #         # batch_preds contains predictions for one batch
        #         # Format depends on model output structure
        #         f.write(f"Batch {batch_idx}:\n")
        #         f.write(f"{batch_preds}\n\n")
        # print(f"Predictions saved to {output_file}")
    else:
        print("start testing")
        res = trainer.test(model, datamodule=data_module)
        with open("test_results.json", "w") as f:
            import json
            json.dump(res, f)
        print("Test results:")
        print(res)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("checkpoint", type=str)
    arg_parser.add_argument("--root_folder_path", type=str, default=None)
    arg_parser.add_argument("--split_folder_name", type=str, default=None)
    arg_parser.add_argument("--batch-size", type=int, default=None)
    arg_parser.add_argument("--save-predictions", action='store_true', default=False)
    arg_parser.add_argument("--output-file", type=str, default="deepwukong_predictions.txt")
    return arg_parser


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    test(__args.checkpoint, __args.root_folder_path,__args.split_folder_name, __args.batch_size, __args.save_predictions, __args.output_file)#, __args.sample_percentage,__args.sub_folder)
