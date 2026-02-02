# Данные

Ожидается UTF-8 CSV со столбцами `product_id`, `seller_id`, `title`, `description`,
`category`. Идентификатор товара должен быть уникален, текст — непустым, а у каждой
категории должно быть не менее трёх продавцов с этой доминирующей категорией: это
необходимо для независимых train/validation/test частей.

`smoke.csv` — полностью синтетический детерминированный набор (seed `42`) для проверки
конвейера; он не предназначен для содержательной оценки качества.

Публичный источник для полноценного эксперимента: [UCI Product Classification and
Clustering](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering),
35311 товарных предложений от 306 продавцов, лицензия CC BY 4.0. При адаптации
`Merchant ID` соответствует `seller_id`, `Product Title` — `title`, `Category Label` —
`category`; `description` можно оставить пустым. Исходный датасет в репозиторий не входит.
