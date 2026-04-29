# Data Contract for Lab Outputs

Единый формат выходных файлов для `lab1` и `lab2`:

- Расположение: `labX/data/*.csv`
- Кодировка: UTF-8
- Разделитель: запятая
- Десятичный разделитель: точка (`InvariantCulture`)

## Схема CSV

Заголовок:

`method,x,approx,exact,abs_error,rel_error`

Поля:

- `method` — имя численного метода (`Euler`, `Heun`, `RungeKutta4`)
- `x` — узел сетки `x_k`
- `approx` — численное значение `y~(x_k)`
- `exact` — аналитическое значение `phi(x_k)`
- `abs_error` — `|phi(x_k) - y~(x_k)|`
- `rel_error` — `|phi(x_k) - y~(x_k)| / |y~(x_k)|` (0, если знаменатель близок к нулю)

## Имена файлов

- `lab1/data/euler.csv`
- `lab1/data/heun.csv`
- `lab1/data/rungekutta4.csv`
- `lab2/data/rungekutta4.csv`
