## mlp_autonomous_vehicle
Manually implemented an MLP and trained a model to enable a self-driving car to reach its destination, with the results presented through an interface.
### 使用語言及技術
Python、Tkinter、類神經網路、機器學習
### MLP架構
```mermaid
graph LR
    A[輸入維度] --> B[4]
    A --> C[6]

    subgraph Model2
				direction LR
        Input[輸入層: 5 神經元]
        Input --> Hidden1[隱藏層 1: 10 神經元]
        Hidden1 --> Hidden2[隱藏層 2: 8 神經元]
        Hidden2 --> Output[輸出層: 1 神經元]
    end

    subgraph Model1
				direction LR
        Input2[輸入層: 3 神經元]
        Input2 --> Hidden21[隱藏層 1: 6 神經元]
        Hidden21 --> Hidden22[隱藏層 2: 10 神經元]
        Hidden22 --> Output2[輸出層: 1 神經元]
    end

    B --> Model1
    C --> Model2
```

### 實作結果
demo: https://www.youtube.com/watch?v=7EDUYxnnots

#### 4維: 前方距離、右方距離、左方距離、方向盤得出角度 
![image](https://github.com/user-attachments/assets/77502ed6-5cdc-4d01-a134-659a89f1d3a8)

#### 6維: X 座標、Y 座標、前方距離、右方距離、左方距離、方向盤得出角度
![image](https://github.com/user-attachments/assets/0132aada-aa69-4b5a-a060-39c23582536a)

