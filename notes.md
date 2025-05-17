### Intro

- O que é aprendizagem por reforço?
Metodologia de aprendizagem de máquina onde um agente aprende a resolver uma determinada tarefa por tentativa e erro.

O agente recebe um estado, proveniente do ambiente, determina uma ação à partir do estado e de sua função de decisão f(estado) -> ação e o ambiente atualiza o estado do agente à partir de sua ação e também fornece uma recompensa.

Com base no estado, na ação escolhida e na recompensa obtida, a função de decisão do agente é atualizada para maximizar a recompensa obtida.

 Exemplificando em alto nível:

```python
state = env.start()
action = agent.predict(state)
new_state, reward = env.step()
agent.learn(reward,state,action)
```
- O que é um ambiente?
Simulação matemática capaz de receber ações de agentes e retornar estados resultantes.

Exemplo:
simulação de formigas em busca de comida, cada formiga é um agente distinto no mesmo ambiente:
<img src="images/full_state.png" alt="environment" width="400"/>

- O que é um estado?
O estado é um vetor que representa as condições do agente em um determinado step ou momento de tempo no ambiente, ele contém medições que o agente é capaz de acessar naquele instante de tempo, com base nas quais o agente decide sua próxima ação.

exemplo de estado, no ambiente acima, cada formiga vê apenas o grid de células 5x5 ao seu redor:
<img src="images/state.png" alt="state" width="300"/>

- O que é uma ação?
Entre um estado e outro, o(s) agente(s) devem selecionar suas ações para aquele step, ou intervalo de tempo.

As ações escolhidas pelos agentes determinam seus estados futuros no ambiente e, consequentemente, suas recompensas.

Uma ação também depende do espaço de ação do agente no ambiente. A título de exemplo, no ambiente em questão, o espaço de ação de cada agente é um vetor de quatro dimensões:

```python
actionSpace = [0 1 2 3]
```
onde cada número corresponde à um sentido de movimentação no ambiente:
<img src="images/meaning.png" alt="action" width="300"/>

Como existem 4 formigas no ambiente, o vetor de ação para cada instante é um dicionário como o seguinte:
```python
{"agent_0":0,"agent_1":3,"agent_2":2,"agent_3":1,}
```
E portanto, para o dicionário acima, cada agente tomaria as seguintes ações e seus estados seguintes seriam resultantes dessas movimentações:
<img src="images/next_states.png" alt="next_states" width="400"/>

- O que são ambientes não deterministicos? Como eles interferem com o aprendizado do agente?

Ambientes não determinísticos adicionam ruído à lógica descrita, uma vez que as ações do agente não determinam inteiramente seus estados futuros e, portanto, também não determinam inteiramente suas recompensas.

Nesse caso se faz necessário uma função de decisão que seja capaz de lidar com esse não determinismo e abstrair quais ações são ideais para determinados estados na média. Um conjunto de dados -> numéro de experiências maior costuma ser necessário e uma função de decisão que aprende de forma mais devagar, mais estável, também costuma ser necessária. 

- O que é uma função de recompensa?

Uma função de recompensa relaciona um determinado estado do agente à um número que reflete a performance do agente na execução da tarefa idealizada.

A título de exemplo, para o ambiente em questão, diferentes funções de recompensa podem ser elaboradas a depender do comportamento que o pesquisador deseja observar nos agentes.

Função 1: Agentes devem cooperar para chegar na recompensa.
```python
# 1 PARA TODOS AGENTES CASO ALGUM TENHA ATINGIDO A RECOMPENSA
# 0 PARA TODOS AGENTES CASO O CONTRÁRIO
def reward(env):
    if any(env.agents.pos == env.target.pos):
        return 100
    return 0
```
O objetivo da função acima seria maximizar a busca dos agentes no ambiente, isto é, como todos ganham recompensa caso 1 encontre o target, eles deveriam aprender que para maximizarem sua recompensa individual deveriam evitar percorrer caminhos já percorridos por outros agentes (indicados pelas células em verde, feromônios) e dessa forma, explorar o ambiente de forma mais eficiente.

Entretanto, é evidente que a função é muito esparsa, principalmente no início do treinamento, os agentes provavelmente demorariam a aprender que o target leva à uma alta recompensa e também possívelmente nem apresentariam o comportamento emergente descrito. Em suma, o aprendizado pode não convergir, ou pode exigir um número altíssimo de observações por parte dos agentes.

Função 2: Alternativa menos sparsa:
Para treinar os agentes de forma mais direcionada ao comportamento descrito, uma segunda função de treinamento poderia ser avaliada:
```python
# 100 PARA TODOS AGENTES CASO ALGUM TENHA ATINGIDO A RECOMPENSA
# -1 PARA TODOS AGENTES CASO ALGUM ESTEJA OCUPANDO UMA CÉLULA COM FEROMÔNIO
# +1 PARA TODOS AGENTES CASO ALGUM ESTEJA OCUPANDO UMA CÉLULA SEM FEROMÔNIO
def reward(env):
    reward_pool = 0
    if any(env.agents.pos == env.target.pos):
        return 100 # algum ja achou a recompensa, retorna 100 e mata o episodio

    for agent in agents:
        if env[agent.pos] == 3: # pheromone cell
            reward_pool -= 1
            continue
        else: # is not in pheromone cell
            reward_pool += 1
    return 1
        
    if 
    return 0
```


A função de recompensa * NÃO DEVE *: fornecer informação excessiva sobre o ambiente que o agente não terá acesso durante sua avaliação em um cenário real. Visto que nesse caso o agente aprenderia a repetir ações de alta recompensa, sem "compreensão" da lógica do ambiente. O agente aprenderia uma heurística que poderia ser implementada em código procedural sem necessidade de aprendizado por reforço.

Exemplo:
Uma função de recompensa que penaliza toda ação que não aproxima o agente da recompensa forneceria informação sobre onde está o target exatamente e não incentiva qualquer tipo de comportamento colaborativo.

Esse tipo de função que simplifica o problema pode ser aplicado em casos onde estamos aplicando ** Transfer learning **, como um passo intermediário no aprendizado do agente. Transfer learning consiste em treinar o agente em várias etapas sequenciais que aumentam em grau de realismo da função de recompensa, com a esperança de que o agente aprenda "conceitos" básicos em relação à sua tarefa para facilitar o aprendizado com a função de recompensa final, que pode ser muito esparsa ou muito complexa.


- Quais são as funções de decisão estudadas?