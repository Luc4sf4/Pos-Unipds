import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};

const WEIGHTS = {
    //item mais importante para recomendar algo
    category: 0.4,
    color: 0.3,
    price: 0.2,
    // item menos importante para recomendar algo
    age: 0.1
};
 const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

 // treina a rede neural para identificar padrões de compra dos usuários e recomendar produtos com base nas idades
 // ou seja, uma pessoa de 25 anos que comprou um produto, a rede neural vai aprender que pessoas de 25 anos tendem
 //  a comprar esse produto e recomendar para outras pessoas de 25 anos
 function makeContext(catalog, users) {
        const ages = users.map(u => u.age);
        const prices = catalog.map(p => p.price);

        const minAge = Math.min(...ages)
        const maxAge = Math.max(...ages)

        const minPrice = Math.min(...prices)
        const maxPrice = Math.max(...prices)

        const colors = [...new Set(catalog.map(p => p.color))];
        const categories = [...new Set(catalog.map(p => p.category))];

        const colorIndex = Object.fromEntries(
            colors.map((color, index) => {
               return [color, index]
            })
        )

        const categoriesIndex = Object.fromEntries(
            categories.map((category, index) => {
               return [category, index]
            })
        )
        //Computar a media de idade dos compradores por produto
        // (ajuda a personalizar)
        const midAge = (minAge + maxAge) / 2;
        const ageSums = {}
        const ageCounts = {}

        users.forEach(user => {
            user.purchases.forEach(p => {
                //Soma de todas idades
                ageSums[p.name] = (ageSums[p.name] || 0) + user.age
                //Soma quantas idades a gente tem
                ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
            })
        });
        // Calcula a média por produto depois de contar todas as compras.
        const productAverageNorm = Object.fromEntries(
            catalog.map(product => {
                const avg = ageCounts[product.name]
                    ? ageSums[product.name] / ageCounts[product.name]
                    : midAge;

                return [product.name, normalize(avg, minAge, maxAge)];
            })
        );

        return {
            catalog,
            users,
            colorIndex,
            categoriesIndex,
            productAverageNorm,
            minAge,
            maxAge,
            minPrice,
            maxPrice,
            numCategories: categories.length,
            numColors: colors.length,
            // price + age + color + category
            dimensions: 2 + categories.length + colors.length,
        }
    }

const oneHotWeighted = (index, length, weight) => tf.oneHot(index, length).cast('float32').mul(weight);
function encodeProduct(product, context) {
    //primeiro, definir os pesos (definidor no inicio do arquivo)
    // normalizar o preço do produto para o intervalo [0, 1] e multiplicar pelo peso definido
    // aplicando o peso para a recomendação do produto com base no preço
    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price,
    ])
    const age = tf.tensor1d([
        (context.productAverageNorm[product.name] ?? 0.5) * WEIGHTS.age
    ])

    //Gera um numero de 0 - 1 baseado na quantidade
    //de categorias que existe dentro de nosso sistema
    //e o nosso peso
    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d([price, age, category, color]);

}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const catalog = await (await fetch('/data/products.json')).json();

    const context = makeContext(catalog, users);
    context.productVectors = catalog.map(product => {
        return {
            name: product.name,
            //quando indentificar a probabilidade esta, ja temos os dados no meta dados
            meta: {...product},
            vector: encodeProduct(product, context).dataSync()
        }
    })
    console.log('Product vectors:', context.productVectors);

    _globalCtx = context;

    postMessage({
        type: workerEvents.trainingLog,
        epoch: 1,
        loss: 1,
        accuracy: 1
    });

    setTimeout(() => {
        postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
        postMessage({ type: workerEvents.trainingComplete });
    }, 1000);


}
function recommend(user, ctx) {
    console.log('will recommend for user:', user)
    // postMessage({
    //     type: workerEvents.recommend,
    //     user,
    //     recommendations: []
    // });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
