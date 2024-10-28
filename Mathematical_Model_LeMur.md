
Notations
---
### Iterators
- $\texttt{m}$ : index for machines
- $\texttt{p}$ : index for products in an order
- $\texttt{c}$ : index for production cycles
- $\texttt{l}$ : index for levate in some cycle

### Sets
- $\texttt{P\_setup = P - \{p in running\_prods and p.current\_op\_type == 0\}}$
- $\texttt{P\_load = P - \{p in running\_prods and p.current\_op\_type <= 1\}}$
- $\texttt{P\_running = P - \{p in running\_prods and p.current\_op\_type <= 2\}}$
- $\texttt{P\_unload = P - \{p in running\_prods and p.current\_op\_type <= 3\}}$


### Inputs
1. $\texttt{max\_levate(p)}$ : number of levate to do in an ordinary cycle for product $\texttt{p}$
2. $\texttt{start\_date(p)}$ : minimum start time for any cycle of $\texttt{p}$
3. $\texttt{due\_date(p)}$ : maximum end time for any cycle of $\texttt{p}$
4. .


### Domains
1. $\texttt{worktime\_domain(p)}$ : domain which excludes :
	- out of working time time steps (night times, holidays)
	- time steps  $< \texttt{start\_date(p)}$
	- time steps $\ge \texttt{due\_date(p)}$
2. .



Decision Variables ( require search )
---

$\texttt{A[p,c,m]}$
- **domain** : $\{0,1\}$
- **description** : keeps track of (product, cycle, machine) assignments

$\texttt{COMPLETE[p,c]}$
- **domain** : $\{0,1\}$
- **description** : cycle $\texttt{c}$ of $\texttt{p}$ is a complete cycle

$\texttt{NUM\_LEVATE[p,c]}$
- **domain** : $\texttt{[0, max\_levate(p)]}$
- **description** : number of levate 

### Time beginnings

$\texttt{SETUP\_BEG[p,c]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : beginning of machine machine START operation

$\texttt{LOAD\_BEG[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : beginning of machine load operation
  
$\texttt{UNLOAD\_BEG[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : beginning of machine unload operation

### Velocity

$\texttt{VELOCITY[p,c]}$ :
- **domain** : $\{-1, 0, 1\}$
- **description** : velocity at which cycle $\texttt{c}$ of product $\texttt{p}$ runs


Other Variables ( no search needed, easily calculated )
---

#### Levata / Cycle activity

$\texttt{ACTIVE\_LEVATA[p,c,l]}$
- **domain** : $\{0,1\}$
- **description** : states if a levata is active (has to be done) or not
- behavior : 
	- $\forall \texttt{.(p,c,l)}$
	  $\texttt{ACTIVE\_LEVATA[p,c,l] == l < NUM\_LEVATE[p,c]}$
		  
$\texttt{ACTIVE\_CYCLE[p,c]}$
- **domain** : $\{0,1\}$
- **description** : states if a cycle is active (has to be done) or not
- behavior : 
	- $\forall \texttt{.(p,c)}$
	  $\texttt{ACTIVE\_CYCLE[p,c] == ACTIVE\_LEVATA[p,c,0]}$

#### Costs

$\texttt{SETUP\_COST[p,c]}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : cost (time) of machine machine setup operation
- **behavior** :
	- $\forall \texttt{.(p in P\_setup, c) : A[p,c,m]} \implies$
	  $\texttt{SETUP\_COST[p,c] == base\_setup\_cost[m,p] + GAP}$
	
$\texttt{LOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : cost (time) of machine machine load operation
- **behavior** :
	- $\forall \texttt{.(p in P\_load,c,l) : A[p,c,m]} \implies$
	  $\texttt{LOAD\_COST[p,c,l] == base\_load\_cost[m,p] + GAP}$

$\texttt{UNLOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : cost (time) of machine machine unload operation 
- **behavior** :
	- $\forall \texttt{.(p in P\_unload,c) : A[p,c,m]} \implies$
	  $\texttt{UNLOAD\_COST[p,c] == base\_unload\_cost[m,p] + GAP}$
	
$\texttt{LEVATA\_COST[p,c]}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : cost (time) of machine machine LEVATA operation 
- **behavior** : 
	-  $\forall\texttt{.(p in P\_running,c) : ACTIVE\_CYCLE[p,c]} \implies$  $\texttt{LEVATA\_COST[p,c] == base\_levata\_cost[p] - VELOCITY[p,c] * velocity\_step[p]}$
	-  $\forall\texttt{.(p in P\_running,c) : ACTIVE\_CYCLE[p,c].Not()} \implies$
	  $\texttt{LEVATA\_COST[p,c] == 0}$

#### Operators

$\texttt{A\_OPERATOR[o,p,c,2]}$
- **domain** : $\{0,1\}$
- **description** : assignment : operator - product - cycle - levata phase (load / unload)
  
$\texttt{OPERATOR\_PER\_GROUP[o]}$
- **domain** : $[1, \texttt{max\_operators}]$
- **description** : number of operators assigned to group $\texttt{o}$


#### Time Ends

$\texttt{SETUP\_END[p,c]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : end of machine machine setup operation
- **behavior** : 
	- $\forall\texttt{.(p,c) : ACTIVE\_CYCLE[p,c]} \implies$
	  $\texttt{LOAD\_END[p,c,l] == LOAD\_BEG[p,c,l] + LOAD\_COST[p,c,l]}$
	  
$\texttt{LOAD\_END[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : end of machine machine load operation
- **behavior** : 
	- $\forall\texttt{.(p,c,l) : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{LOAD\_END[p,c,l] == LOAD\_BEG[p,c,l] + LOAD\_COST[p,c,l]}$
	  
$\texttt{UNLOAD\_END[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : end of machine machine unload operation
- **behavior** : 
	- $\forall\texttt{.(p,c,l) : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{UNLOAD\_END[p,c,l] == UNLOAD\_BEG[p,c,l] + UNLOAD\_COST[p,c,l]}$
	
$\texttt{CYCLE\_END[p,c]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : time at which the cycle ends
- **behavior** : 
	- $\forall\texttt{.(p,c) : ACTIVE\_CYCLE[p,c]} \implies$
	  $\texttt{CYCLE\_END[p,c] == UNLOAD\_END[p,c,-1]}$

### Production goal

$\texttt{KG\_CYCLE[p,c]}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : Kg produced by cycle $\texttt{c}$ of product $\texttt{p}$
- **behavior** : 
	- $\forall\texttt{.(p,c,m) :}$
	  $\texttt{KG\_CYCLE[p,c] == NUM\_LEVATE[p,c] * }   \sum_{\texttt{m}} \texttt{(A[p,c,m] * Kg\_per\_levata[m,p])}$

Constraints (search space reduction related)
---

SEARCH REDUCTION :
1. $\texttt{COMPLETE[p,c]} \implies \texttt{NUM\_LEVATE[p,c] == max\_levate(p)}$
2. $\texttt{COMPLETE[p,c]} \implies \texttt{ACTIVE\_CYCLE[p,c]}$ 
3. $\texttt{ACTIVE\_CYCLE[p,c]} \implies \texttt{NUM\_LEVATE[p,c] == 0}$ 
4. $\texttt{ACTIVE\_CYCLE[p,c] and COMPLETE[p,c].Not()} \implies \texttt{NUM\_LEVATE[p,c] > 0}$

COMPACTNESS :
1. $\texttt{COMPLETE[p,c]} \ge \texttt{COMPLETE[p,c+1]}$
2. $\texttt{ACTIVE\_CYCLE[p,c]} \ge \texttt{ACTIVE\_CYCLE[p,c+1]}$


Constraint (LeMur specific)
---

1. A cycle can be assigned to 1 or zero machine
	- $\forall \texttt{.(p,c)}$
	  $\texttt{AtMostOne} (\sum_l \texttt{A[p,c,:]} )$
	  
2. At most one partial cycle per product 
	- $\forall \texttt{.p}$
	  $\sum_c({\texttt{ACTIVE\_CYCLE[p,c]})} - \sum_c(\texttt{COMPLETE[p,c]}) \le 1$
	  
3. Start date / Due date : (can be defined at domain level)
	- $\forall \texttt{.p not in running\_prods}$
	  $\texttt{SETUP\_BEG[p,c]} \ge \texttt{start\_date(p)}$
	- $\forall \texttt{.p}$
	  $\texttt{UNLOAD\_END[p,c,-1]} < \texttt{due\_date(p)}$

4. Objective : all products must reach the requested production
	- $\forall \texttt{.p}$
	  $\texttt{Kg\_requested(p)} \le \sum_{\texttt{c}}( \texttt{KG\_CYCLE[p,c]}) < \texttt{Kg\_requested(p) + best\_machine(p)}$
	
5. Define ordering between time variables 
	1. LOAD
	- $\forall \texttt{.(p in P\_load, c, l == 0) : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{LOAD\_BEG[p,c,l]} \ge \texttt{SETUP\_END[p,c])}$
	- $\forall \texttt{.(p in P\_load, c, l > 0) : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{LOAD\_BEG[p,c,l] == UNLOAD\_END[p,c,l-1])}$
			  
	2. UNLOAD
	- $\forall \texttt{.(p in P\_unload, c, l) : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{UNLOAD\_BEG[p,c,l]} \ge \texttt{LOAD\_END[p,c,l] + LEVATA\_COST[p,c]}$
		  
	3. PARTIAL LOADS / UNLOADS : 
	- $\texttt{ACTIVE\_LEVATA[p,c,l].Not() and ACTIVE\_CYCLE[p,c]} \implies$
		- $\forall \texttt{.(p in P\_load, c, l > 0)}$
			- $\texttt{LOAD\_BEG[p,c,l] == LOAD\_BEG[p,c,l-1]}$
		- $\forall \texttt{.(p in P\_unload, c, l > 0)}$
			- $\texttt{UNLOAD\_BEG[p,c,l] == UNLOAD\_BEG[p,c,l-1]}$
		  
	4. INACTIVE CYCLES
	- $\forall \texttt{.(p,c,l) : ACTIVE\_CYCLE[p,c].Not()} \implies$
		- $\texttt{LOAD\_BEG[p,c,l] == 0}$
		- $\texttt{UNLOAD\_BEG[p,c,l] == 0}$
  
6. No overlap between product cycles on same machine :
- $\forall \texttt{.(m,p,c)}$
  $\texttt{NoOverlap( [ }\texttt{A[p,c,m]} \implies \exists \texttt{.Interval(SETUP\_BEG[p,c], UNLOAD\_END[p,c,-1]) ] )}$

7. Operators constraints
- $\forall \texttt{.(o,p,c) : ACTIVE\_CYCLE[p,c]} \implies$
$\texttt{ExactlyOne(A\_OPERATORS[o,p,c,0]) and ExactlyOne(A\_OPERATORS[o,p,c,1])}$  $\texttt{NoOverlap(}\texttt{A\_OPERATORS[o,p,c,0]} \implies \exists \texttt{.Interval(LOAD\_BEG[p,c,l], LOAD\_END[p,c,l])}$
$\texttt{NoOverlap(}\texttt{A\_OPERATORS[o,p,c,1]} \implies \exists \texttt{.Interval(UNLOAD\_BEG[p,c,l], UNLOAD\_END[p,c,l])}$


8. Handle initialization of running products.
	- $\texttt{running\_prod[p]}$ : contains specific information needed for products already running on some machine when the scheduling starts
		- $\texttt{.machine}$ : machine associated to $\texttt{p}$
		- $\texttt{.velocity}$ : velocity at which machine associated to $\texttt{p}$ is running
		- $\texttt{.remaining\_levate}$ : levate $\texttt{p}$ needs to do in order to complete the current executing cycle
		- $\texttt{.current\_op\_type}$ : states in which phase is $\texttt{p}$ when the scheduling starts, can have 4 possible values $\{0,1,2,3\}$ associated to a all cycle's possible phases => {setup, load, running, unload}
		- $\texttt{.remaining\_time}$ : remaining time needed to complete $\texttt{.current\_op\_type}$ operation

- $\forall \texttt{.p} \in \texttt{running\_products}$ :
  notation : $\texttt{p.}$ stands for $\texttt{runnig\_prod[p].}$ (in this section)

	1. Fix machine assignment, velocity and number of levate
		 $\texttt{A[p, 0, p.machine] == 1}$
		 $\texttt{VELOCITY[p,0] == p.velocity}$
		 $\texttt{NUM\_LEVATE[p,0] == p.remaining\_levate}$
	  
	2. SETUP phase
		- $\texttt{if p.current\_op\_type == 0}$
		  $\texttt{SETUP\_BEG[p, 0] == 0}$
		  $\texttt{SETUP\_COST[p,0] == p.remaining\_time + GAP}$
	  
	3. LOAD phase
		- $\texttt{if p.current\_op\_type == 1}$
		  $\texttt{SETUP\_BEG[p, 0] == 0 and LOAD\_BEG[p, 0] == 0}$
		  $\texttt{LOAD\_COST[p,0,0] == p.remaining\_time + GAP}$

	4. RUNNING phase
	-  $\texttt{if p.current\_op\_type == 2}$
	  $\texttt{SETUP\_BEG[p, 0] == 0 and LOAD\_END[p, 0] == 0}$
	  $\texttt{LEVATA\_COST[p,0,0] == p.remaining\_time + VELOCITY[p,0] * velocity\_step[p]}$

	5. UNLOAD phase
		- $\texttt{if p.current\_op\_type == 3}$
		  $\texttt{SETUP\_BEG[p, 0] == 0 and UNLOAD\_BEG[p, 0] == 0}$
		  $\texttt{UNLOAD\_COST[p,0,0] == p.remaining\_time + GAP}$






