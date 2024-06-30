## Neural-network-for-sentiment-analysis

# Введение
В этом учебном проекте я использовал полносвязанную нейронную сеть для классификации тональности (положительный/отрицательный) отзывов о фильмах из IMDb.

# Данные
Для обучения модели я буду использовать классический датасет IMDb, который содержит 50 000 отзывов, каждый из которых помечен как "положительный" или "отрицательный".

# Модель
Я буду использовать простую полносвязанную нейронную сеть с одним скрытым слоем. Сеть будет принимать вектор слов, закодированный с помощью one hot encoding, представляющий отзыв, и выводить вероятность того, что отзыв является "положительным" или "отрицательным".

# Результаты
В итоге получилась модель, которая с точностью около 85%, может классифицировать тональность отзывов о фильмах.
