import tf from '@tensorflow/tfjs-node';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    //primeira camada da rede 
    // entrada de 7 posicao (idade normalizada + 3 cores +  3 localizacoes)

    // 80 neurônios = colocado esse numero por conta da pouca base de treinos
    // + neuronios =  + complexidade para a rede aprender 
    // que significa mais processamento ela vai usar

    //a RelU age como um filtro 
    // ela deixa somente os dados interessantes
    // se a info chegada no neuronio for positiva passa pra frente 
    // caso for nula ou negativa, pode jogar fora, nao vale nada
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // Saida: 3 neuronios, 1 pra cada categoria
    // sendo elas (premium, medium, basic)

    //Activation: softmax que normaliza a saida em probabilidades
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    //Complizando o modelo usando o Optimazer: Adam (Adaptive Momente Estimation) 
    // um treinador pessoal moderno para redes neurais
    // ajustando os pesos de forma efificiente e inteligente
    // aprendendo com erros e acertos

    //loss: categoricalCrossentropy
    // compara o que o modelo "acha" (os scores de catagoria) com a resposta certa

    // quanto mais distante da previsão correta, maior o erro(loss), e mais o modelo vai aprender
    //Exemplo: classificacao de imagens, recomendacao e categorizacao de usuario
    // qualquer coisa em que a resposta certa seria "apenas um entre varias possiveis"
    model.compile({ optimizer: 'adam', 
        loss: 'categoricalCrossentropy', 
        metrics: ['accuracy'] });
    

    //Treinando o modelo
    //verbose: desabilite o log interno (so se usa o callback)
    //epochs: quantidade de vezes que vai rodar no dataset
    // shuffle: embaralha os dados para evitar vies
    await model.fit(
        inputXs, 
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            // callbacks: {
            //     onEpochEnd: (epoch, logs) => {
            //         console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            //     }
            // }
        }
    )

    return model;
}
async function predict(model, pessoaTensorNormalizado) {
    //transformar o array JS para o tensor do tensorflow
    const tfInput = tf.tensor2d([pessoaTensorNormalizado]);

    //fazendo a previsão (output de um vetor de 3 probabilidades)
    const pred = model.predict(tfInput);
    const predArray = await pred.array();
    // console.log(`Predição: ${predArray}`);
    return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

const model = await trainModel(inputXs, outputYs);

const pessoa = { nome: "João", idade: 28, cor: "verde", localizacao: "São Paulo" };

// Normalizando os dados de entrada para a pessoa a ser classificada
// exemplo: idade normalizada = (idade - idade_min) / (idade_max - idade_min)

const pessoaTensorNormalizado = [
    0.2, // idade normalizada
    1,   // azul
    0,   // vermelho
    0,   // verde
    0,   // São Paulo
    1,   // Rio
    0    // Curitiba
]

const predictions = await predict(model, pessoaTensorNormalizado);
const result = predictions
.sort((a, b) => b.prob - a.prob)
.map(p => `${labelsNomes[p.index]}: ${(p.prob * 100).toFixed(2)}%`)
.join('\n ');
console.log(result);// Pegando a categoria com maior probabilidade