# SCHEDULER for JSP - LP approach
This is an WORK IN PROGRESS VERSION.

## -- Tasks --

### TO DO
- Define the general structure on the *README* for understanding and navigate the scheduler file

- Populate the lib folder with methods and data structures for improving the code clarity
### TO ANALYZE
- Change the orders "dict" data structure in to a list
- Change the `cycles` attribute name in to `velata` inside `products` paramter
- Change the `products` attribute name inside `orders` parameter for conflicting with the `products` parameter
## -- Guide --

## Structure

### Optimization

For the optimization, the `Branch and Bound` algorithm has been chosen.

## Costraints and Parameters

### Costraints

### Parameters

` machines = ['M1', 'M2'] `: list of all possible machines(not divided in the 3 categories)

`orders` : orders dict where each entry is a client order
- `due_date` : the maximum time available for completing the order, expressed in hours
- `products` : list of the products to schedule production for that order

`products` : products available to production dict, where are specified all the operations to be made for producing them.

