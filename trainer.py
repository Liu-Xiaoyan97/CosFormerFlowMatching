from transformers import Trainer
from CosFormer.configuration_LLMFCosformer import LLMFCosformerConfig
from CosFormer.cosformer import LLFMCosformerForFlowMatching
from utils import (
    load_training_args_from_yaml, 
    MyDataCollator, 
    MyDataset, 
    MyTrainer,
    DetailedProgressCallback
)


if __name__ == "__main__":
    
    
    config = LLMFCosformerConfig()
    model = LLFMCosformerForFlowMatching(config)
    data_collator = MyDataCollator()
    train_dataset = MyDataset(tokenizer_path="Tokenizer_32768_v1", dataset_name="stanfordnlp/imdb", split="train", chunk_size=512)
    test_dataset = MyDataset(tokenizer_path="Tokenizer_32768_v1", dataset_name="stanfordnlp/imdb", split="test", chunk_size=512)
    training_args = load_training_args_from_yaml("config/trainingargs.yml")
    
    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[DetailedProgressCallback()],
    )

    # 7. 开始训练
    print("Starting training...")
    trainer.train()
    
    # 8. 保存最终模型
    trainer.save_model()
    print("Training completed and model saved.")