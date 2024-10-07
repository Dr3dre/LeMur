# SCHEDULER for JSP - LP approach

## Structure



## Costraints and Parameters

### Costraints

``

### Parameters

` machines = ['M1', 'M2'] `: list of all possible machines(not divided in the 3 categories)

`orders` : orders dict where each entry is a client order
- `due_date` : the maximum time available for completing the order, expressed in hours
- `prducts` : list of the products to schedule production for that order

`products` 

## Optimization

For the optimization, the `Branch and Bound` algorithm has been chosen.