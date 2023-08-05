# instruction of usage

# what you can do through this project
+ ## can operate spike rate encoding
+ ## can use back porpagate calculation thorough spiking neural network
+ ## can classify iris classification by SNN

# 1. image of spike encoding

## 1-1 . Encoding
+ ### since Iris data is real based value, the process convert to time-series data is required.  
![image](https://github.com/GTAKAGI/PSNN/assets/114473358/45354d11-0aa2-4eb5-a16b-735e24dc6366)

## 1-2 . Structure of encoding inputs
+ ## input static data are converted into time-series data
1. input data (static) 4×150
2. input data (spikes) 4×100×150
※ 100 is total time



![image](https://github.com/GTAKAGI/PSNN/assets/114473358/09e9da5c-1586-4a5f-8e69-efed0b1a27eb)
![image](https://github.com/GTAKAGI/PSNN/assets/114473358/9c7358b4-3aef-4183-b0d6-efd3e76a0cc1)

# 2. input data to SNN
+ ### network architecture is shown below.  4 × 3 × 3
![image](https://github.com/GTAKAGI/PSNN/assets/114473358/0bd27ac5-ed9c-4d76-8b05-ff818262149e)
# 3. how to evaluate
+ ### the input labels are given 0 to 3
+ ### setosa : 0, virginica : 1, virgicolor : 2
+ ### Thsoe correspond to last neurons

