# Prob1 Code

## 文件解释

[main.py](./main.py) 

里面有`train`和`test`两个函数，对task1-3进行训练和测试

[task1.py](task1.py) 完成第一小问的代码
[task2.py](task2.py) 完成第二小问的代码
[task3.py](task3.py) 完成第三小问的代码

[utils.py](./utils.py) 辅助函数

[model.py](./model.py) 支持向量机的模型
[dataloader.py](./dataloader.py) 数据输入处理

[try_SMOTE.py](./try_SMOTE.py) 一个简单的尝试，可以忽略


## 运行

```js
python main.py --mode test --out_dir ./test --model_dir ./model/rand_seed_14

解释：
mode: "train" or "test", default "train"
out_dir: <yout out_dir 相对于"./output">, default datetime.now().strftime("%m%d_%H%M%S")
model_dir: <your model_dir>,仅mode为test时生效,default ./model/rand_seed_14
```
