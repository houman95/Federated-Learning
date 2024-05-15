# Federated Learning Project w/ DLR friends
## Description
This code simulates FL in a random access channel.

Line 57 is where you try differnet sparsification budgets. (Remember to change line 483 and 485 while simulating)

Line 63 is where you try different number of slots. (Remember to change line 420 while simulating)

Line 405 is where you try different gamma momentum. (Remember to change line 488 while simulating)

## Use the following code and run it in .ipynb file to run mainVGG16.py
```python
  %run mainVGG16.py --R 1 --M 0 --P 60
```

## Use the following code and run it in .ipynb file to run sim_to_find_opt_activeUsers.py
```python
  %run sim_to_find_opt_activeUsers.py
```
