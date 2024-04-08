# MulCo
The code of our paper.

## Requirements
- Python >=3.6
- PyTorch >=1.9

---
main.py
  >This is main function. After running the code, you should see a text file with the results saved in the corresponding directory. The results will have seven columns: epoch number, training accuracy, test accuracy, top 5.

## Running

python main.py --multiprocessing-distributed

**Methods and models**

In main.py, specify the method argument to choose one of the 3 losses available:
- ComSifted
- ComSoft
- ComWeight
