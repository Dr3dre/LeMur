
Notations
---
### Iterators
- $\texttt{m}$ : index for machines
- $\texttt{p}$ : index for products in an order
- $\texttt{c}$ : index for production cycles
- $\texttt{l}$ : index for levate in some cycle

### Sets
- $\texttt{running\_products}$ : products which are already running a machine
- $\texttt{Setup} = \forall \texttt{.(p,c)} - \texttt{(\{p in running\_prods and p.current\_op\_type == 0\}, 0)}$
- $\texttt{Load} = \forall \texttt{.(p,c,l)} - \texttt{(\{p in running\_prods and p.current\_op\_type <= 1\}, 0, 0)}$
- $\texttt{Running } = \forall \texttt{.(p,c,l)} - \texttt{(\{p in running\_prods and p.current\_op\_type <= 2\}, 0, 0)}$
- $\texttt{Unload} = \forall \texttt{.(p,c,l)} - \texttt{(\{p in running\_prods and p.current\_op\_type <= 3\}, 0, 0)}$
 
### Inputs
1. $\texttt{standard\_levate(p)}$ : number of levate to do in an ordinary cycle for product $\texttt{p}$
2. $\texttt{start\_date(p)}$ : minimum start time for any cycle of $\texttt{p}$
3. $\texttt{due\_date(p)}$ : maximum end time for any cycle of $\texttt{p}$
4. $\texttt{number\_of\_operator\_groups}$ : number of operator groups
5. $\texttt{operators\_per\_group}$ : number of operators in a single group
6. $\texttt{Kg\_requested(p)}$ : Kg to produce of product $\texttt{p}$
7. $\texttt{start\_shift}$ :
8. $\texttt{end\_shift}$ : 


### Domains
1. $\texttt{worktime\_domain(p)}$ : domain which excludes :
	- out of working time time steps (night times, holidays)
	- time steps  $< \texttt{start\_date(p)}$
	- time steps $\ge \texttt{due\_date(p)}$
2. .


### Derived Constants
1. $\texttt{best\_kg\_cycle(p)}$ : Kg produced in a single cycle by the better performing machine for product $\texttt{p}$ 
2. $\texttt{gaps\_per\_day[g]}$ : width of gap associated to day $\texttt{g}$
3. .


Decision Variables ( require search )
---

$\texttt{A[p,c,m]}$
- **domain** : $\{0,1\}$
- **description** : keeps track of (product, cycle, machine) assignments

$\texttt{COMPLETE[p,c]}$
- **domain** : $\{0,1\}$
- **description** : cycle $\texttt{c}$ of $\texttt{p}$ is a complete cycle

$\texttt{NUM\_LEVATE[p,c]}$
- **domain** : $\texttt{[0, standard\_levate(p)]}$
- **description** : number of levate cycle $\texttt{c}$ of $\texttt{p}$ needs to do

### Time beginnings

$\texttt{SETUP\_BEG[p,c]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : beginning of machine setup operation (set velocity, load elastomero, ...)

$\texttt{LOAD\_BEG[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : beginning of machine load operation (load fusi)
  
$\texttt{UNLOAD\_BEG[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : beginning of machine unload operation (unload fusi)

### Velocity

$\texttt{VELOCITY[p,c]}$ :
- **domain** : $\{-1, 0, 1\}$
- **description** : velocity at which cycle $\texttt{c}$ of product $\texttt{p}$ runs

### Operators

$\texttt{A\_OP\_SETUP[o,p,c]}$
- **domain** : $\{0,1\}$
- **description** : assignment : operator - product - cycle
$\texttt{A\_OP[o,p,c,l,\{0,1\}]}$
- **domain** : $\{0,1\}$
- **description** : assignment : operator - product - cycle - levata phase (load / unload)
  
Other Variables ( no search needed, easily calculated )
---

#### Levata / Cycle activity

$\texttt{ACTIVE\_LEVATA[p,c,l]}$
- **domain** : $\{0,1\}$
- **description** : states if a levata is active (it exists) or not
- **scope** : allows to handle existence of partial cycles 
- **behavior** : 
	- $\forall \texttt{.(p,c,l)}$
	  $\texttt{ACTIVE\_LEVATA[p,c,l] == (l < NUM\_LEVATE[p,c])}$
	
$\texttt{ACTIVE\_CYCLE[p,c]}$
- **domain** : $\{0,1\}$
- **description** : states if a cycle is active (it exists) or not
- **scope** : just for readability purposes 
- **behavior** : 
	- $\forall \texttt{.(p,c)}$
	  $\texttt{ACTIVE\_CYCLE[p,c] == ACTIVE\_LEVATA[p,c,0]}$

$\texttt{PARTIAL\_CYCLE[p,c]}$
- **domain** : $\{0,1\}$
- **description** : 
- **scope** : just for readability purposes 
- **behavior** : 
	- $\forall \texttt{.(p,c)}$
	  $\texttt{PARTIAL\_CYCLE[p,c] == COMPLETE\_CYCLE[p,c].Not() and ACTIVE\_CYCLE[p,c]}$

#### Costs

$\texttt{GAP\_AT[G]}$
- Element variable => $\texttt{gaps\_at(G) == gaps\_per\_day[g]}$
- **description** :

$\texttt{GAP(begin, base\_cost)}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : 
- **behavior** : 
	- $\texttt{G = begin // 24}$
	- $\texttt{UB = end\_shift + 24*(G-1)}$
	- $\texttt{NEEDS\_GAP == begin + base\_cost > UB }$
		- $\texttt{NEEDS\_GAP} \implies \texttt{GAP == GAP\_AT[G]}$
		- $\texttt{NEEDS\_GAP.Not()} \implies \texttt{GAP == 0}$


$\texttt{SETUP\_COST[p,c]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine setup operation
- **behavior** :
	- $\forall \texttt{.(p,c) in Setup : A[p,c,m]} \implies$
	  $\texttt{SETUP\_COST[p,c] == base\_setup\_cost[m,p] + GAP(begin,base\_cost)}$
	
$\texttt{LOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine load operation
- **behavior** :
	- $\forall \texttt{.(p,c,l) in Load : A[p,c,m]} \implies$
	  $\texttt{LOAD\_COST[p,c,l] == base\_load\_cost[m,p] + GAP(begin,base\_cost)}$

$\texttt{UNLOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine unload operation 
- **behavior** :
	- $\forall \texttt{.(p,c,l) in Unload : A[p,c,m]} \implies$
	  $\texttt{UNLOAD\_COST[p,c] == base\_unload\_cost[m,p] + GAP(begin,base\_cost)}$
	
$\texttt{LEVATA\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine levata operation 
- **behavior** : 
	-  $\forall\texttt{.(p,c,l) in Running : ACTIVE\_LEVATA[p,c,l]} \implies$  $\texttt{LEVATA\_COST[p,c,l] == base\_levata\_cost[p] - VELOCITY[p,c] * velocity\_step[p]}$


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

### Operators

$\texttt{OPERATOR\_PER\_GROUP[o]}$
- **domain** : $[1, \texttt{max\_operators}]$
- **description** : number of operators assigned to group $\texttt{o}$
- **behavior** : 
	- $\forall \texttt{.(o)}$ : 
	  $\texttt{OPERATOS\_PER\_GROUP[o] == operators\_per\_group}$

### Production goal

$\texttt{KG\_CYCLE[p,c]}$
- **domain** : $[0, \texttt{best\_kg\_cycle(p)}]$
- **description** : Kg produced by cycle $\texttt{c}$ of product $\texttt{p}$
- **behavior** : 
	- $\forall\texttt{.(p,c,m) :}$
	  $\texttt{KG\_CYCLE[p,c] == NUM\_LEVATE[p,c] * }   \sum_{\texttt{m}} \texttt{(A[p,c,m] * Kg\_per\_levata[m,p])}$

Constraints (search space reduction related)
---

### Search Reduction :
1. $\texttt{A[p,c,m]} \implies \texttt{ACTIVE\_CYCLE[p,c]}$ ?
2. $\texttt{COMPLETE[p,c]} \iff \texttt{NUM\_LEVATE[p,c] == standard\_levate(p)}$
3. $\texttt{COMPLETE[p,c] or PARTIAL\_CYCLE[p,c]} \iff \texttt{ACTIVE\_CYCLE[p,c]}$ 
4. $\texttt{PARTIAL\_CYCLE[p,c]} \iff \texttt{0 < NUM\_LEVATE[p,c] < standard\_levate(p) }$
5. $\texttt{ACTIVE\_CYCLE[p,c]} \implies \texttt{}$

### Compactness :
1. $\texttt{COMPLETE[p,c]} \ge \texttt{COMPLETE[p,c+1]}$
2. $\texttt{ACTIVE\_CYCLE[p,c]} \ge \texttt{ACTIVE\_CYCLE[p,c+1]}$

Constraint (LeMur specific)
---

1. A cycle can be assigned to 1 or 0 machine
	- $\forall \texttt{.(p,c)}$
	  $\texttt{AtMostOne} (\texttt{A[p,c,:]} )$
	  
2. At most one partial cycle per product 
	- $\forall \texttt{.p}$
	  $\texttt{AtMostOne(PARTIAL\_CYCLE[p,:])}$
	  
3. Start date / Due date : (is actually defined at domain level)
	- $\forall \texttt{.p not in running\_prods}$
	  $\texttt{SETUP\_BEG[p,c]} \ge \texttt{start\_date(p)}$
	- $\forall \texttt{.p}$
	  $\texttt{UNLOAD\_END[p,c,-1]} < \texttt{due\_date(p)}$

4. Objective : all products must reach the requested production
	- $\forall \texttt{.p}$
	  $\texttt{Kg\_requested(p)} \le \sum_{\texttt{c}}( \texttt{KG\_CYCLE[p,c]}) < \texttt{Kg\_requested(p) + best\_kg\_cycle(p)}$
	
5. Define ordering between time variables 
	1. LOAD
	- $\forall \texttt{.(p,c,l) in Load, if l == 0 : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{LOAD\_BEG[p,c,l]} \ge \texttt{SETUP\_END[p,c])}$
	- $\forall \texttt{.(p,c,l) in Load, if l > 0 : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{LOAD\_BEG[p,c,l] == UNLOAD\_END[p,c,l-1])}$
			  
	2. UNLOAD
	- $\forall \texttt{.(p,c,l) in Unload : ACTIVE\_LEVATA[p,c,l]} \implies$
	  $\texttt{UNLOAD\_BEG[p,c,l]} \ge \texttt{LOAD\_END[p,c,l] + LEVATA\_COST[p,c,l]}$
		  
	3. PARTIAL LOADS / UNLOADS : 
	- $\texttt{PARTIAL\_CYCLE[p,c] and ACTIVE\_LEVATA[p,c,l].Not()} \implies$
		- $\forall \texttt{.(p,c,l) in Load, if l > 0 :}$
			- $\texttt{LOAD\_BEG[p,c,l] == LOAD\_BEG[p,c,l-1]}$
		- $\forall \texttt{.(p,c,l) in Unoad, if l > 0 :}$
			- $\texttt{UNLOAD\_BEG[p,c,l] == UNLOAD\_BEG[p,c,l-1]}$
		  
	4. INACTIVE CYCLES (?)
	- $\forall \texttt{.(p,c,l) : ACTIVE\_CYCLE[p,c].Not()} \implies$
		- $\texttt{LOAD\_BEG[p,c,l] == 0}$
		- $\texttt{UNLOAD\_BEG[p,c,l] == 0}$
  
6. No overlap between product cycles on same machine :
- $\forall \texttt{.(m)}$
	- $\forall \texttt{.(p,c)}$$\texttt{NoOverlap([}\texttt{A[p,c,m]} \implies \exists \texttt{.Interval(SETUP\_BEG[p,c], UNLOAD\_END[p,c,-1])])}$

7. Operators constraints
- $\forall \texttt{.(o) } \forall \texttt{.(p,c) : ACTIVE\_CYCLE[p,c]} \implies$
  $\texttt{ExactlyOne(A\_OP\_SETUP[:,p,c])}$
  - $\forall \texttt{.(o) } \forall \texttt{.(p,c,l,t in \{0,1\}) : ACTIVE\_LEVATA[p,c,l]} \implies$
    $\texttt{ExactlyOne(A\_OP[:,p,c,l,t])}$
- $\forall \texttt{.(o) } \forall \texttt{.(p,c)}$
$\texttt{NoOverlap(}\texttt{A\_OP\_SETUP[o,p,c]} \implies \exists \texttt{.Interval(SETUP\_BEG[p,c], SETUP\_END[p,c])}$
- $\forall \texttt{.(o) } \forall \texttt{.(p,c,l)}$
$\texttt{NoOverlap(}\texttt{A\_OP[o,p,c,l,0]} \implies \exists \texttt{.Interval(LOAD\_BEG[p,c,l], LOAD\_END[p,c,l])}$
$\texttt{NoOverlap(}\texttt{A\_OP[o,p,c,l,1]} \implies \exists \texttt{.Interval(UNLOAD\_BEG[p,c,l], UNLOAD\_END[p,c,l])}$


8. Handle initialization of running products.
	- $\texttt{running\_prod[p]}$ : contains specific information needed for products already running on some machine when the scheduling starts
		- $\texttt{.machine}$ : machine associated to $\texttt{p}$
		- $\texttt{.operator}$ : operator associated to $\texttt{p}$
		- $\texttt{.velocity}$ : velocity at which machine associated to $\texttt{p}$ is running
		- $\texttt{.remaining\_levate}$ : levate $\texttt{p}$ needs to do in order to complete the current executing cycle
		- $\texttt{.current\_op\_type}$ : states in which phase is $\texttt{p}$ when the scheduling starts, can have 4 possible values $\{0,1,2,3\}$ associated to a all cycle's possible phases => {setup, load, running, unload}
		- $\texttt{.remaining\_time}$ : remaining time needed to complete $\texttt{.current\_op\_type}$ operation, this also include the relative gap as it can be pre computed

- $\forall \texttt{.p} \in \texttt{running\_products}$ :
  notation : $\texttt{p.}$ stands for $\texttt{runnig\_prod[p].}$ (in this section)

	1. Fix machine assignment, velocity and number of levate
		 $\texttt{A[p,0,p.machine] == 1}$
		 $\texttt{VELOCITY[p,0] == p.velocity}$
		 $\texttt{NUM\_LEVATE[p,0] == p.remaining\_levate}$
	  
	2. SETUP phase
		- $\texttt{if p.current\_op\_type == 0}$
		  $\texttt{SETUP\_BEG[p,0] == 0}$
		  $\texttt{SETUP\_COST[p,0] == p.remaining\_time}$
		  ---
		  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
	  
	3. LOAD phase
		- $\texttt{if p.current\_op\_type == 1}$
		  $\texttt{SETUP\_BEG[p,0] == 0 and LOAD\_BEG[p,0,0] == 0}$
		  $\texttt{LOAD\_COST[p,0,0] == p.remaining\_time}$
		  --
		  $\texttt{SETUP\_COST[p,0] == 0}$
		  --
		  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
		  $\texttt{A\_OP[p.operator,p,0,0,0] == 1}$

	4. RUNNING phase
	-  $\texttt{if p.current\_op\_type == 2}$
	  $\texttt{SETUP\_BEG[p,0] == 0}$
	  $\texttt{LOAD\_BEG[p,0,0] == 0}$
	  --
	  $\texttt{LEVATA\_COST[p,0,0] == p.remaining\_time + VELOCITY[p,0] * velocity\_step[p]}$
	  --
	  $\texttt{SETUP\_COST[p,0] == 0}$
	  $\texttt{LOAD\_COST[p,0,0] == 0}$
	  --
	  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
	  $\texttt{A\_OP[p.operator,p,0,0,0] == 1}$

	5. UNLOAD phase
		- $\texttt{if p.current\_op\_type == 3}$
		  $\texttt{SETUP\_BEG[p,0] == 0}$
		  $\texttt{LOAD\_BEG[p,0,0] == 0}$
		  $\texttt{UNLOAD\_BEG[p,0,0] == 0}$
		  ---
		  $\texttt{SETUP\_COST[p,0] == 0}$
		  $\texttt{LOAD\_COST[p,0,0] == 0}$
		  $\texttt{LEVATA\_COST[p,0,0] == 0}$
		  ---
		  $\texttt{UNLOAD\_COST[p,0,0] == p.remaining\_time}$
		  ---
		  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
		  $\texttt{A\_OP[p.operator,p,0,0,0] == 1}$
		  $\texttt{A\_OP[p.operator,p,0,0,1] == 1}$
