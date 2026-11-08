# CORD v2 data summary

Dataset: `naver-clova-ix/cord-v2`

| role | split | receipts | lines | words |
|---|---|---|---|---|
| work | train | 800 | 10987 | 19367 |
| gold | validation | 100 | 1219 | 2186 |
| eval | test | 100 | 1309 | 2356 |

## Label space (29 categories, derived from the work split)

| category | train | validation | test |
|---|---|---|---|
| `menu.cnt` | 1903 | 210 | 220 |
| `menu.discountprice` | 97 | 4 | 10 |
| `menu.etc` | 6 | 0 | 3 |
| `menu.itemsubtotal` | 1 | 0 | 6 |
| `menu.nm` | 2100 | 222 | 251 |
| `menu.num` | 94 | 4 | 11 |
| `menu.price` | 2093 | 220 | 246 |
| `menu.sub.cnt` | 142 | 32 | 17 |
| `menu.sub.nm` | 319 | 49 | 36 |
| `menu.sub.price` | 123 | 14 | 20 |
| `menu.sub.unitprice` | 14 | 0 | 0 |
| `menu.unitprice` | 621 | 50 | 67 |
| `menu.vatyn` | 2 | 1 | 0 |
| `sub_total.discount_price` | 68 | 6 | 7 |
| `sub_total.etc` | 112 | 2 | 13 |
| `sub_total.othersvc_price` | 2 | 0 | 0 |
| `sub_total.service_price` | 98 | 13 | 12 |
| `sub_total.subtotal_price` | 541 | 67 | 66 |
| `sub_total.tax_price` | 362 | 47 | 45 |
| `total.cashprice` | 526 | 64 | 69 |
| `total.changeprice` | 502 | 64 | 56 |
| `total.creditcardprice` | 124 | 15 | 17 |
| `total.emoneyprice` | 47 | 4 | 2 |
| `total.menuqty_cnt` | 233 | 26 | 29 |
| `total.menutype_cnt` | 43 | 3 | 7 |
| `total.total_etc` | 28 | 4 | 3 |
| `total.total_price` | 784 | 98 | 96 |
| `void_menu.nm` | 1 | 0 | 0 |
| `void_menu.price` | 1 | 0 | 0 |

Categories outside the label space: none
