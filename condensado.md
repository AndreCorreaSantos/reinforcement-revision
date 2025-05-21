

#### O que é Aprendizagem por Reforço (RL)?
A **aprendizagem por reforço** é uma metodologia de aprendizado de máquina onde um **agente** aprende a resolver tarefas por tentativa e erro, interagindo com um **ambiente**. O agente observa um **estado**, toma uma **ação** baseada em sua função de decisão e recebe uma **recompensa** e um novo estado do ambiente. O objetivo é ajustar essa função para maximizar a recompensa acumulada.

- **Ambiente**: Simulação que processa ações do agente e retorna estados e recompensas.  
  *Exemplo*: Um grid onde formigas buscam comida.
- **Estado**: Vetor que descreve as condições atuais do agente.  
  *Exemplo*: Um grid 5x5 ao redor de uma formiga.
- **Ação**: Decisão do agente que altera o estado e a recompensa.  
  *Exemplo*: Mover-se em uma direção (cima, baixo, esquerda, direita).
- **Ambientes não determinísticos**: Introduzem incerteza, pois ações nem sempre levam a resultados previsíveis, exigindo funções de decisão robustas e mais dados para aprendizado estável.
- **Função de recompensa**: Associa estados a valores que refletem a performance do agente.  
  *Exemplo*: +100 se uma formiga acha comida, -1 por repetir caminhos.

#### Algoritmos de RL

1. **Q-Learning**  
   - Método tabular que mapeia pares estado-ação a **valores Q** (expectativa de recompensa futura).  
   - Atualiza Q com base na recompensa e no melhor Q futuro:  
     *Equação*: \( Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)] \).  
   - **Hiperparâmetros**:  
     - **Alpha (α)**: Taxa de aprendizado.  
     - **Gamma (γ)**: Peso de recompensas futuras.  
     - **Epsilon (ε)**: Exploração vs. explotação.

2. **SARSA**  
   - Similar ao Q-Learning, mas atualiza Q com base na ação realmente tomada (on-policy).  
   - Ideal para ambientes estocásticos.  
   - *Equação*: \( Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)] \).

3. **Deep Q Learning (DQN)**  
   - Usa uma **rede neural** para estimar Q em espaços de estado grandes.  
   - **Experience replay**: Armazena experiências e treina com amostras aleatórias, reduzindo overfitting.

4. **Reinforce**  
   - Algoritmo **policy-based** que otimiza a política diretamente com uma rede neural, gerando uma distribuição de probabilidade sobre ações.

5. **Actor Critic**  
   - Combina um **ator** (escolhe ações via política) e um **crítico** (avalia ações via valores).  
   - O crítico reduz variância e estabiliza o aprendizado.

6. **PPO (Proximal Policy Optimization)**  
   - Algoritmo actor-critic com **clipping** na função de perda, limitando atualizações bruscas na política para maior estabilidade.

