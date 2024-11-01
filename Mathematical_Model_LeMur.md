
Notations
---
### Iterators
- $\texttt{m}$ : index for machines
- $\texttt{p}$ : index for products in an order
- $\texttt{c}$ : index for production cycles
- $\texttt{l}$ : index for levate in some cycle

### Sets
1. $\texttt{running\_products}$ : products which are already running a machine
2. sets of indices relative to running products which need to be omitted to some constraints and treated separately 
	- $\texttt{SETUP\_EXCL} = \texttt{(\{p in running\_prods and p.current\_op\_type == 0\}, 0)}$
	- $\texttt{LOAD\_EXCL} = \texttt{(\{p in running\_prods and p.current\_op\_type <= 1\}, 0, 0)}$
	- $\texttt{LEVATA\_EXCL} = \texttt{(\{p in running\_prods and p.current\_op\_type <= 2\}, 0, 0)}$
	- $\texttt{UNLOAD\_EXCL} = \texttt{(\{p in running\_prods and p.current\_op\_type <= 3\}, 0, 0)}$
 
### Inputs
1. $\texttt{start\_shift}$ : start time of working shift
2. $\texttt{end\_shift}$ : end time of working shift
3. $\texttt{time\_units\_per\_day}$ : defines quantization scale of scheduling operation
   (24 : hours, 48 : half hours, ... , 1440 : minutes )
   --
4. $\texttt{start\_date(p)}$ : minimum start time for any cycle of $\texttt{p}$
5. $\texttt{due\_date(p)}$ : maximum end time for any cycle of $\texttt{p}$
   --
6. $\texttt{Kg\_requested(p)}$ : Kg to produce of product $\texttt{p}$
7. $\texttt{standard\_levate(p)}$ : number of levate to do in an ordinary cycle for product $\texttt{p}$
   --
8. $\texttt{number\_of\_operator\_groups}$ : number of operator groups
9. $\texttt{operators\_per\_group}$ : number of operators in a single group
10. .


### Domains
1. $\texttt{worktime\_domain(p)}$ : domain which excludes :
	- out of working time time steps (night times, holidays)
	- time steps  $< \texttt{start\_date(p)}$
	- time steps $\ge \texttt{due\_date(p)}$
2. .


### Derived Constants
1. $\texttt{best\_kg\_cycle(p)}$ : Kg produced in a single cycle by the better performing machine for product $\texttt{p}$ 
2. $\texttt{gap\_at\_day[g]}$ : width of gap associated to day $\texttt{g}$
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

$\texttt{PARTIAL\_CYCLE[p,c]}$
- **domain** : $\{0,1\}$
- **description** : 
- **scope** : just for readability purposes 
- **behavior** : 
	- $\forall \texttt{.(p,c)}$
	  $\texttt{PARTIAL\_CYCLE[p,c] == COMPLETE\_CYCLE[p,c].Not() and ACTIVE\_CYCLE[p,c]}$

#### Costs

$\texttt{BASE\_SETUP\_COST[p,c]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : base cost (time) of machine setup operation, it accounts for machine association & available operators
- **behavior** :
	- $\forall \texttt{.(p,c) not in SETUP\_EXCL : A[p,c,m]} \implies$
	  $\texttt{BASE\_SETUP\_COST[p,c] == base\_setup\_cost[m,p]}$
	- $\forall \texttt{.(p,c) not in SETUP\_EXCL : ACTIVE\_CYCLE[p,c].Not()} \implies$
	  $\texttt{BASE\_SETUP\_COST[p,c] == 0}$
	  
$\texttt{BASE\_LOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : base cost (time) of machine load operation, it accounts for machine association & available operators
- **behavior** :
	- $\forall \texttt{.(p,c,l) not in LOAD\_EXCL : A[p,c,m]} \implies$
	  $\texttt{BASE\_LOAD\_COST[p,c,l] == base\_load\_cost[m,p]}$
	- $\forall \texttt{.(p,c) not in LOAD\_EXCL : ACTIVE\_LEVATA[p,c,l].Not()} \implies$
	  $\texttt{BASE\_LOAD\_COST[p,c] == 0}$
	  
$\texttt{BASE\_UNLOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : base cost (time) of machine unload operation, it accounts for machine association & available operators
- **behavior** :
	- $\forall \texttt{.(p,c,l) not in UNLOAD\_EXCL : A[p,c,m]} \implies$
	  $\texttt{BASE\_UNLOAD\_COST[p,c,l] == base\_unload\_cost[m,p]}$
	- $\forall \texttt{.(p,c) not in UNLOAD\_EXCL : ACTIVE\_LEVATA[p,c,l].Not()} \implies$
	  $\texttt{BASE\_UNLOAD\_COST[p,c] == 0}$

$\texttt{function make\_gap\_var(BEGIN, BASE\_COST, IS\_ACTIVE)}$
- **domain** : $[0, \texttt{horizon}]$
- **description** : 
- **behavior** : 
	- $\texttt{gap\_at\_day[G] == GAP\_SIZE}$
	- $\texttt{G = begin // time\_units\_per\_day}$
	- $\texttt{UB = end\_shift + time\_units\_per\_day * G}$
	- $\texttt{NEEDS\_GAP == begin + base\_cost > UB }$
		- $\texttt{NEEDS\_GAP and IS\_ACTIVE} \implies \texttt{GAP == GAP\_SIZE}$
		- $\texttt{NEEDS\_GAP.Not() and IS\_ACTIVE} \implies \texttt{GAP == 0}$
		- $\texttt{IS\_ACTIVE.Not()} \implies \texttt{GAP == 0}$
- **returns** : $\texttt{GAP}$

$\texttt{SETUP\_COST[p,c]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine setup operation
- **behavior** :
	- $\forall \texttt{.(p,c) not in SETUP\_EXCL :}$
	  $\texttt{GAP = make\_gap\_var(SETUP\_BEG, BASE\_SETUP\_COST, ACTIVE\_CYCLE)}$
	  $\texttt{SETUP\_COST[p,c] == BASE\_SETUP\_COST[p,c] + GAP}$
$\texttt{LOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine load operation
- **behavior** :
	- $\forall \texttt{.(p,c,l) not in LOAD\_EXCL :}$
	  $\texttt{GAP = make\_gap\_var(LOAD\_BEG, BASE\_LOAD\_COST, ACTIVE\_LEVATA)}$
	  $\texttt{LOAD\_COST[p,c,l] == BASE\_LOAD\_COST[p,c,l] + GAP}$
$\texttt{UNLOAD\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine unload operation 
- **behavior** :
	- $\forall \texttt{.(p,c,l) not in UNLOAD\_EXCL :}$
	  $\texttt{GAP = make\_gap\_var(UNLOAD\_BEG, BASE\_UNLOAD\_COST, ACTIVE\_LEVATA)}$
	  $\texttt{UNLOAD\_COST[p,c,l] == BASE\_UNLOAD\_COST[p,c,l] + GAP}$
	
$\texttt{LEVATA\_COST[p,c,l]}$
- **domain** : $[0, \texttt{due\_date(p)}]$
- **description** : cost (time) of machine levata operation 
- **behavior** : 
	-  $\forall\texttt{.(p,c,l) not in LEVATA\_EXCL : ACTIVE\_LEVATA[p,c,l]} \implies$  $\texttt{LEVATA\_COST[p,c,l] == base\_levata\_cost[p] - VELOCITY[p,c] * velocity\_step[p]}$
	- $\forall\texttt{.(p,c,l) not in LEVATA\_EXCL : ACTIVE\_LEVATA[p,c,l].Not()} \implies$
	  $\texttt{LEVATA\_COST[p,c,l] == 0}$


#### Time Ends

$\texttt{SETUP\_END[p,c]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : end of machine machine setup operation
- **behavior** : 
	- $\forall\texttt{.(p,c) :}$
	  $\texttt{LOAD\_END[p,c,l] == LOAD\_BEG[p,c,l] + LOAD\_COST[p,c,l]}$
	  
$\texttt{LOAD\_END[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : end of machine machine load operation
- **behavior** : 
	- $\forall\texttt{.(p,c,l) :}$
	  $\texttt{LOAD\_END[p,c,l] == LOAD\_BEG[p,c,l] + LOAD\_COST[p,c,l]}$
	  
$\texttt{UNLOAD\_END[p,c,l]}$
- **domain** : $x \in \texttt{worktime\_domain(p)}$
- **description** : end of machine machine unload operation
- **behavior** : 
	- $\forall\texttt{.(p,c,l) :}$
	  $\texttt{UNLOAD\_END[p,c,l] == UNLOAD\_BEG[p,c,l] + UNLOAD\_COST[p,c,l]}$

### Operators

$\texttt{OPERATOR\_PER\_GROUP[o]}$
- **domain** : $[2, \texttt{max\_operators}]$
- **description** : number of operators assigned to group $\texttt{o}$ It cannot be less than 2, otherwise it gives problems with the formulation of gap!
- **behavior** : 
	- $\forall \texttt{.(o)}$ : 
	  $\texttt{OPERATOS\_PER\_GROUP[o] == operators\_per\_group}$

### Production goal

$\texttt{KG\_CYCLE[p,c]}$
- **domain** : $[0, \texttt{production\_request(p) + best\_kg\_cycle(p)}]$
- **description** : Kg produced by cycle $\texttt{c}$ of product $\texttt{p}$
- **behavior** : 
	- $\forall\texttt{.(p,c,m) :}$
	  $\texttt{KG\_CYCLE[p,c] == NUM\_LEVATE[p,c] * }   \sum_{\texttt{m}} \texttt{(A[p,c,m] * Kg\_per\_levata[m,p])}$

Aliases
---

$\texttt{ACTIVE\_CYCLE[p,c]}$
- **description** : Alias indicating if a cycle is active
- **behavior** : $\forall \texttt{.(p,c) : ACTIVE\_CYCLE[p,c] = ACTIVE\_LEVATA[p,c,0]}$

$\texttt{CYCLE\_BEG[p,c]}$
- **description** : Alias indicating when the cycle begins
- **behavior** : $\forall \texttt{.(p,c) : CYCLE\_BEG[p,c] = SETUP\_BEG[p,0]}$
$\texttt{CYCLE\_END[p,c]}$
- **description** : Alias indicating when the cycle ends
- **behavior** : $\forall \texttt{.(p,c) : CYCLE\_END[p,c] = UNLOAD\_END[p,c,-1]}$


Constraints (search space reduction related)
---
##### Compactness :
1. $\forall \texttt{.(p,c) : COMPLETE[p,c]} \ge \texttt{COMPLETE[p,c+1]}$
2. $\forall \texttt{.(p,c) : ACTIVE\_CYCLE[p,c]} \ge \texttt{ACTIVE\_CYCLE[p,c+1]}$

Constraint (LeMur specific)
---

1. Cycle machine assignment:
	- An active cycle must have one and only one machine assigned
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{ACTIVE\_CYCLE[p,c]} \implies \texttt{XOR(A[p,c,:])}$
	- A non active cycle must have 0 machines assigned
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{ACTIVE\_CYCLE[p,c].Not()} \implies \texttt{OR(A[p,c,:]).Not()}$

2. At most one partial cycle per product 
	- $\forall \texttt{.(p) :}$
	  $\texttt{AtMostOne(PARTIAL\_CYCLE[p,:])}$

3. Connect cycle specific variables:
	- The complete cycles must be active (only implication to allow for partials)
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{COMPLETE[p,c]} \implies \texttt{ACTIVE\_CYCLE[p,c]}$
	- The partial cycle is the active but not complete (this carries the at most one from partial to active so it needs to be a if and only if)
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{PARTIAL\_CYCLE[p,c]} \iff \texttt{ACTIVE\_CYCLE[p,c] and COMPLETE[p,c].Not()}$

3. Tie number of levate to cycles
	- If the cycle is complete, then the number of levate is the maximum one
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{COMPLETE[p,c]} \iff \texttt{NUM\_LEVATE[p,c] == standard\_levate(p)}$
	- If the cycle is not active the number of levate is 0
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{ACTIVE\_CYCLE[p,c].Not()} \iff \texttt{NUM\_LEVATE[p,c] == 0}$
	- If partial, then we search for the number of levate
		- $\forall \texttt{.(p,c) :}$
		  $\texttt{PARTIAL\_CYCLE[p,c]} \iff \texttt{0 < NUM\_LEVATE[p,c] < standard\_levate(p)}$

4. Start date / Due date : (can actually be defined at domain level)
	- $\forall \texttt{.(p) not in running\_prods :}$
	  $\texttt{SETUP\_BEG[p,c]} \ge \texttt{start\_date(p)}$
	- $\forall \texttt{.(p) :}$
	  $\texttt{UNLOAD\_END[p,c,-1]} < \texttt{due\_date(p)}$

5. Objective : all products must reach the requested production
	- $\forall \texttt{.(p) :}$
	  $\texttt{Kg\_requested(p)} \le \sum_{\texttt{c}}( \texttt{KG\_CYCLE[p,c]}) < \texttt{Kg\_requested(p) + best\_kg\_cycle(p)}$
	
6. Define ordering between time variables 
	1. LOAD
		- $\forall \texttt{.(p,c,l) not in LOAD\_EXCL, if l == 0 : ACTIVE\_LEVATA[p,c,l]} \implies$
		  $\texttt{LOAD\_BEG[p,c,l]} \ge \texttt{SETUP\_END[p,c])}$
		- $\forall \texttt{.(p,c,l) not in LOAD\_EXCL, if l > 0 : ACTIVE\_LEVATA[p,c,l]} \implies$
		  $\texttt{LOAD\_BEG[p,c,l] == UNLOAD\_END[p,c,l-1])}$
  
	2. UNLOAD
		- $\forall \texttt{.(p,c,l) not in UNLOAD\_EXCL : ACTIVE\_LEVATA[p,c,l]} \implies$
		  $\texttt{UNLOAD\_BEG[p,c,l]} \ge \texttt{LOAD\_END[p,c,l] + LEVATA\_COST[p,c,l]}$
		  
	3. PARTIAL LOADS / UNLOADS :  Collapse all non used levata operations at the end of the previous one
		- $\texttt{PARTIAL\_CYCLE[p,c] and ACTIVE\_LEVATA[p,c,l].Not()} \implies$
			- $\forall \texttt{.(p,c,l) not in LOAD\_EXCL, if l > 0 :}$
				- $\texttt{LOAD\_BEG[p,c,l] == LOAD\_BEG[p,c,l-1]}$
			- $\forall \texttt{.(p,c,l) not in UNLOAD\_EXCL, if l > 0 :}$
				- $\texttt{UNLOAD\_BEG[p,c,l] == UNLOAD\_BEG[p,c,l-1]}$
		  
	4. INACTIVE CYCLES they are handled by shoving them at the beginning
		- $\forall \texttt{.(p,c,l) : ACTIVE\_CYCLE[p,c].Not()} \implies$
			- $\texttt{LOAD\_BEG[p,c,l] == 0 ; LOAD\_END[p,c,l] == 0}$
			- $\texttt{UNLOAD\_BEG[p,c,l] == 0 ; UNLOAD\_END[p,c,l] == 0}$
	
7. No overlap between product cycles on same machine :
	- $\forall \texttt{.(m)}$
		- $\forall \texttt{.(p,c)}$ $\texttt{NoOverlap([}\texttt{A[p,c,m]} \implies \exists \texttt{.Interval(SETUP\_BEG[p,c], UNLOAD\_END[p,c,-1])])}$

8. Operators constraints
	- The active cycles' setups must be assigned to one operator $\forall \texttt{.(o) } \forall \texttt{.(p,c) : ACTIVE\_CYCLE[p,c]} \implies \texttt{ExactlyOne(A\_OP\_SETUP[:,p,c])}$
	- The non active cycles' setups must have no operator assigned $\forall \texttt{.(o) } \forall \texttt{.(p,c) : ACTIVE\_CYCLE[p,c].Not()} \implies \texttt{OR(A\_OP\_SETUP[:,p,c]).Not()}$
	- The levate must have an operator assigned for the load and the unload operation:$\forall \texttt{.(o) } \forall \texttt{.(p,c,l,t in \{0,1\}) : ACTIVE\_LEVATA[p,c,l]} \implies \texttt{ExactlyOne(A\_OP[:,p,c,l,t])}$
	- The non active levate must have no operator assigned for the load and the unload operation:$\forall \texttt{.(o) } \forall \texttt{.(p,c,l,t in \{0,1\}) : ACTIVE\_LEVATA[p,c,l]} \implies \texttt{OR(A\_OP[:,p,c,l,t]).Not()}$
	- We create intervals for each operation operators have to handle:<br>$\forall \texttt{.(o,p,c)}$<br> $\texttt{A\_OP\_SETUP[o,p,c]} \implies \exists \texttt{.Interval(SETUP\_BEG[p,c], SETUP\_END[p,c])}$<br>$\forall \texttt{.(o,p,c,l)}$<br>$\texttt{A\_OP[o,p,c,l,0]} \implies \exists \texttt{.Interval(LOAD\_BEG[p,c,l], LOAD\_END[p,c,l]}$<br>$\texttt{A\_OP[o,p,c,l,1]} \implies \exists \texttt{.Interval(UNLOAD\_BEG[p,c,l], UNLOAD\_END[p,c,l]}$
	- No overlap between all of these interval for an operator<br>$\forall\texttt{I} \in \texttt{Intervals. NoOverlap(I)}$


9. Handle initialization of running products.
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
		  - Set Beggining of operations
		  $\texttt{SETUP\_BEG[p,0] == 0}$
		  - Current Operation Cost:
		  $\texttt{SETUP\_COST[p,0] == p.remaining\_time}$
		  - Operator Assignments:
		  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
	  
	3. LOAD phase
		- $\texttt{if p.current\_op\_type == 1}$
		  - Set Beggining of operations
		  $\texttt{SETUP\_BEG[p,0] == 0 and LOAD\_BEG[p,0,0] == 0}$
		  - Current Operation Cost:
		   $\texttt{LOAD\_COST[p,0,0] == p.remaining\_time}$
		   - Previous Operations have 0 cost:
		   $\texttt{SETUP\_COST[p,0] == 0}$
		   Operator Assignments:
		  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
		  $\texttt{A\_OP[p.operator,p,0,0,0] == 1}$

	4. RUNNING phase
		- $\texttt{if p.current\_op\_type == 2}$
			- Set Beggining of operations
			$\texttt{SETUP\_BEG[p,0] == 0}$
			$\texttt{LOAD\_BEG[p,0,0] == 0}$
			- Current Operation Cost:
			$\texttt{LEVATA\_COST[p,0,0] == p.remaining\_time + VELOCITY[p,0] * velocity\_step[p]}$
			- Previous Operations have 0 cost:
			$\texttt{SETUP\_COST[p,0] == 0}$
			$\texttt{LOAD\_COST[p,0,0] == 0}$
			- Operator Assignments:
			$\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
			$\texttt{A\_OP[p.operator,p,0,0,0] == 1}$

	5. UNLOAD phase
		- $\texttt{if p.current\_op\_type == 3}$
		  - Set Beggining of operations
		  $\texttt{SETUP\_BEG[p,0] == 0}$
		  $\texttt{LOAD\_BEG[p,0,0] == 0}$
		  $\texttt{UNLOAD\_BEG[p,0,0] == 0}$
		  - Previous Operations have 0 cost:
		  $\texttt{SETUP\_COST[p,0] == 0}$
		  $\texttt{LOAD\_COST[p,0,0] == 0}$
		  $\texttt{LEVATA\_COST[p,0,0] == 0}$
		  - Current Operation Cost:
		  $\texttt{UNLOAD\_COST[p,0,0] == p.remaining\_time}$
		  - Operator Assignments:
		  $\texttt{A\_OP\_SETUP[p.operator,p,0] == 1}$
		  $\texttt{A\_OP[p.operator,p,0,0,0] == 1}$
		  $\texttt{A\_OP[p.operator,p,0,0,1] == 1}$
