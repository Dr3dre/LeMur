# SCHEDULER for JSP - OR-Tools approach
This is an WORK IN PROGRESS VERSION.

## -- Tasks --

### TO DO
- Define the general structure on the *README* for understanding and navigate the scheduler file

- Populate the lib folder with methods and data structures for improving the code clarity

- fix the operations disconnection during the production scheduling. Insert a costraint on the operation that require certain operations to be made consecutively on the same machine
### TO ANALYZE
- Change the orders "dict" data structure in to a list

## -- Guide --

## Structure - Top-Down View

1. **Orders**
1. **Products**
1. **Cycles**
1. **Operations**


### Optimization

For the optimization, the `Branch and Bound` algorithm has been chosen.

## Costraints and Variables

### Costraints

1.
1.  

### Variables

1. **Shifts**
1. **Operators**

## Parameters

### Global 
` MACHINES = ['M1', 'M2'] `: list of all possible machines(not divided in the 3 categories)

`ORDERS` : orders dict where each entry is a client order
- `due_date` : the maximum time available for completing the order, expressed in hours
- `products` : list of the products to schedule production for that order

`PRODUCTS` : products available to production dict, where are specified all the operations to be made for producing them.

`operation_specifications` : dict of technical attributes of every possible operations  
- `machines` : list of machines where is possible to do this operations
- `base_time` : operation duration *minimum* time
- `requires_supervision` : boolean attribute that indicates the need of having a human supervision/intervetion for this kind of operation.
### ...
