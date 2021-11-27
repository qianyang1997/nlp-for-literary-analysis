import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.config.experimental import list_physical_devices, \
                                           set_memory_growth
from transformers import BertTokenizerFast, \
                         TFBertForSequenceClassification, \
                         TFTrainer, \
                         TFTrainingArguments
from sklearn.metrics import classification_report
from preprocessing import create_train_test


# tensorflow config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = list_physical_devices('GPU')
if physical_devices:
    set_memory_growth(physical_devices[0], True)


class Bert:

    def __init__(self, model_name, token_name=None, train=True,
                 maxlen=50, epochs=10, decay=0.01):
        
        self.maxlen = maxlen
        
        if train:
            
            token_name = model_name
            
            self.training_args = TFTrainingArguments(
                num_train_epochs=epochs,
                per_device_train_batch_size=8,
                weight_decay=decay,
                load_best_model_at_end=True,
                logging_steps=200,
                evaluation_strategy="steps",
                output_dir="./output",
                save_steps=400
            )
            
            with self.training_args.strategy.scope():
                self.model = TFBertForSequenceClassification.from_pretrained(model_name)
                
        else:
            self.model = TFBertForSequenceClassification.from_pretrained(model_name)
            
        self.tokenizer = BertTokenizerFast.from_pretrained(token_name,
                                                           do_lower_case=True)

    def encode(self, text, labels=None):
        if labels is None:
            encodings = self.tokenizer(text, truncation=True, padding='max_length',
                                       max_length=self.maxlen, return_tensors='tf')
        else:
            encodings = self.tokenizer(text, truncation=True, padding='max_length',
                                       max_length=self.maxlen)         
            encodings = tf.data.Dataset.from_tensor_slices((dict(encodings), 
                                                            labels))
        return encodings

    def train(self, train_dataset, test_dataset):

        trainer = TFTrainer(model=self.model,
                            args=self.training_args,
                            train_dataset=train_dataset,
                            eval_dataset=test_dataset)
        trainer.train()
        return trainer

    def predict(self, text_list):
        input_text = self.encode(text_list)
        result = self.model(input_text)[0].numpy()
        prediction_prob = tf.nn.softmax(result).numpy()
        prediction_hard = np.argmax(prediction_prob, axis=1)
        return prediction_prob, prediction_hard
    
    def save_model(self, model_path):
        self.model.save_pretrained(model_path)
        
    def save_result(self, x_test, y_test):
        pred_prob, pred_hard = self.predict(x_test)
        df1 = pd.DataFrame({'text': x_test, 'label': y_test, 'pred_hard': pred_hard})
        df2 = pd.DataFrame(pred_prob)
        result = pd.concat([df1, df2], axis=1)
        # print test set evaluation metrics
        print(classification_report(result.label, result.pred_hard))
        return result
        
    def main(self, x_train, x_test, y_train, y_test):
        train_encoded = self.encode(x_train, y_train)
        test_encoded = self.encode(x_test, y_test)
        trainer = self.train(train_encoded, test_encoded) 
        print('Finished')
        print(self.evaluate(trainer))


if __name__ == '__main__':
    model_name = "bert-base-uncased"
    trained_model = "test_model"
    example_text = "Why can't I like waffles?"
    
    #x_train, x_test, y_train, y_test = create_train_test('data/barack.txt',
                                                         #'data/michelle.txt',
                                                         #0.2, 0.2, 42, 21)
    
    # train new
    #pip = Bert(model_name, maxlen=100, epochs=100)
    #self.main(x_train, x_test, y_train, y_test)
    
    # save model
    #pip.save_model('test_model')
    
    # load trained model
    pip = Bert(trained_model, model_name, train=False)
    print(pd.DataFrame(pip.predict(["values", "morals", "America", "united",
             "immigrant", "president", "they", "Oval Office",
             "wealth", "law", "healthcare", "insightful",
             "communicate", "speech", "military", "fellow",
             "dream", "private", "idea", "Kant", "income",
             "inequality", "ambition", "poverty", "diplomacy",
             "angry", "terror", "war", "compromise", "enforcement",
             "leadership", "economy", "analyze", "think", "impartial",
             "legislation", "tax", "growth", "idealistic",
             "politics", "employment", "fair"])))
    #print(pip.predict([example_text]))
    
    # save result
    #result = pip.save_result(x_test, y_test)
    #result.to_csv('test_set_performance.csv', index=False)