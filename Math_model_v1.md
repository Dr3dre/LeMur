notation

- $\texttt{m}$ : index for machines
- $\texttt{p}$ : index for products in an order
- $\texttt{c}$ : index for production cycles
- $\texttt{l}$ : index for levate in some cycle

Decision Variables ( require search )
---

MACHINE - PRODUCT - CYCLE - LEVATA assignments 
- $\texttt{Y[m,p,c,l]}$
	- **type** : Boolean
	- **domain** : $\{0,1\}$
	- **description** : keeps track of machine - product - cycle - levata assignments

START PROCEDURE
- $\texttt{S\_beg[p,c]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>beginning</u> of machine machine START operation
MACHINE LOAD PROCEDURE
- $\texttt{ML\_beg[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>beginning</u> of machine machine LOAD operation
MACHINE UNLOAD PROCEDURE
- $\texttt{MU\_beg[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>beginning</u> of machine machine UN-LOAD operation

VELOCITY
- $\texttt{V[p,c]}$ :
	- **type** : Integer
	- **domain** : $\{-1, 0, 1\}$
	- **description** : VELOCITY at which cycle $\texttt{c}$ of product $\texttt{p}$ runs




Other Variables ( no search needed, easily calculated )
---

COMPLETION PROCEDURE

(?)
- $\texttt{C\_beg[p,c]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_beginning\_domain}$
	- **description** : <u>beginning</u> of machine machine COMPLETION operation
- $\texttt{C\_cost[m,p]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine COMPLETION operation


COSTS RELATED
- $\texttt{S\_cost[p,c]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine START operation
- $\texttt{ML\_cost[p,c,l]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine LOAD operation
- $\texttt{MU\_cost[p,c,l]}$
	- **type** : Integer
	- **domain** : $[0, horizon]$
	- **description** : <u>cost</u> of machine machine UN-LOAD operation 

END OF PROCEDURES
- $\texttt{S\_end[p,c]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_end\_domain}$
	- **description** : <u>end</u> of machine machine START operation
- $\texttt{C\_end[p,c]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_end\_domain}$
	- **description** : <u>end</u> of machine machine COMPLETION operation
- $\texttt{MU\_end[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_end\_domain}$
	- **description** : <u>end</u> of machine machine UN-LOAD operation
- $\texttt{ML\_end[p,c,l]}$
	- **type** : Integer
	- **domain** : $x \in \texttt{work-time\_end\_domain}$
	- **description** : <u>end</u> of machine machine LOAD operation


PRODUCT ON MACHINE
- $\texttt{ON[m,p]}$
	- **type** : Bool
	- **domain** : $\{0,1\}$
	- **description** : product $\texttt{p}$ is on machine $\texttt{m}$
	- **behavior** :  $$\texttt{ON[m,p] == BoolOr(Y[m,p,:,:])}$$

COMPLETE / PARTIAL CYCLES
- $\texttt{X[m,p,c]}$
	- **type** : Bool
	- **domain** : $\{0,1\}$
	- **description** : true if cycle $\texttt{c}$ of product $\texttt{p}$ is a <u>partial cycle</u> (all levate in the cycle are done), false if it's a <u>complete cycle</u> (not all levate are done)
	- **behavior** : $$
	 \begin{cases}
	  \texttt{( BoolXor(Y[m,p,c,:]) ).OnlyEnforceIf(X[m,p,c], ON[m,p])} \\ 
	  \texttt{( BoolAnd(Y[m,p,c,:]) ).OnlyEnforceIf(X[m,p,c].Not(), ON[m,p])}
	  \end{cases}
	  $$
	- **note** : $\texttt{levate\_per\_cycle(p)}$ is an input constant variable defining how many levate there are in a cycle for a certain product $\texttt{p}$


NUMBER OF LEVATE
- $\texttt{N\_lev[m,p]}$
	- **type** : Integer
	- **domain** : $\texttt{[0, max\_levate(p]}$
	- **description** : number of levate done by a certain product $\texttt{p}$ on some machine $\texttt{m}$
	- **behavior** : $$\texttt{N\_lev[m,p] == sum(Y[m,p,:,:])}$$

Constraints
---

1. A levata can be in only one or zero machines  $$\texttt{AtMostOne( Y[:,p,c,l] )}$$
	- avoids same levata to be assigned to multiple machines
	- allow allocation of partial cycles


2. Al levate of a specific cycle must be on the same machine  $$
  \texttt{AtMostOne(X[:,p,c])}
  $$
3. Regulate amount of partial cycles to up to $1$ per product $$\texttt{AtMostOne(X[:,p,:])}$$
4. Reach the requested amount of Kg for the product$$
   \texttt{production[m,p] = Kg\_per\_levata[m,p]} \cdot \texttt{N\_lev[m,p]}
   $$
$$
   \texttt{LinearExpression( sum(production[:,p]), lb=Kg\_Request(p)), ub=upper\_bound(p) )}
   $$
- $\texttt{upper\_bound(p) = Kg\_Request(p) + Kg\_per\_levata[m,p]}$
	- where $\texttt{m ==}$ best machine


5. Start / Due date$$
   \begin{cases}
   \texttt{S\_beg[p,c]} \ge \texttt{start\_date[p]} \\
   \texttt{C\_end[p,c]} \le \texttt{start\_date[p]}
   \end{cases}
   $$
6. Assigning Load / Unload time 
   $$
   \begin{cases}
   \texttt{ML\_beg[p,c,l]} \ge \texttt{S\_end[p,c]} & \text{if} \  l=0 \\
   \texttt{ML\_beg[p,c,l]} \ge  \texttt{MU\_end[p,c,l-1]} & \text{otherwise}
   \end{cases}
   $$
    $$
   \texttt{MU[p,c,l] == [p,c]}
   $$
   


DUMP
---

1. Levate are ordered Right-Most (right tightness of $\texttt{Y}$ assignments)$$\texttt{Y[m,p,c,l-1]} \le \texttt{Y[m,p,c,l]} $$
	- if at least one levata is assigned it can be found in the <u>last position</u>
		- $ex$ : $\texttt{Y[m,p,c,levate\_per\_cycle(p)-1] == 1}$
	- last levata of a cycle can easily be accessed (good for future constraints)
	-  search space is smaller  
