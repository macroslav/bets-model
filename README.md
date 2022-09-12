# Bets-model

ML-часть сервиса для 

## Частично реализованы: 

+ Пайплайн для обучения и процессинга данных
+ Модуль с трансформером данных
+ Модуль с моделями
+ Модуль для оценки эффективности моделей

## В планах:
+ нагенерить фичи
+ построить модели первого уровня:
  + линейные;
  + бустинги на разных наборах фичей + разные архитектуры;
  + нейронные сети (tabnet, перцептрон);
+ модели второго уровня
+ тюнинг параметров
+ придумать алгоритм для выбора событий

Набор признаков:
- Страна, лига, сезон, дата, время
- Коэффициенты
- Стоимость составов команды, средний возраст команды, кол-во легионеров, группировка по позициям (стоимость, возраст и т.д.)
- Стадионы, города
- Тренера (имя, национальность, возраст, кол-во времени на посту)
- Ассистенты  (имя, национальность, возраст, кол-во времени на посту)
- Трансферы (считаются за прошлое зимнее и за прошлое летнее лето)
- Составы команд - 15 игроков (имя, национальность, возраст, дата прихода в команду, стоимость, бывшая команда)
- Травмы игроков (а также стоимость состава без учета травмированных игроков)
- Статистика игроков за прошлый сезон (матчи, голы, передачи, кол-во сыгранных минут)
- Статистика команды за прошлый сезон (победы, ничьи, поражения, голы)
- Статистика команды за текущий сезон (победы, ничьи, поражения, голы)