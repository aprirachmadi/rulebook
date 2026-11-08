---
name: cord-guideline
description: The receipt labeling guideline, version v1. Label definitions and examples.
---
# Receipt labeling guideline v1

Give every receipt line exactly one label from the list below. A line can hold a key word and
a value together (for example `TOTAL 45,500`); the whole line gets one label.

Examples show the line text and, in brackets, other lines on the same row of the receipt.

## 1. Menu items

### 1.1 `menu.nm`
The name of a purchased item.
Example: `WHOLE WHEAT PAN BREAD` [same row: `1`, `20,000`]

### 1.2 `menu.num`
An item code or number printed with the item (for example a product code).
Example: none yet.

### 1.3 `menu.cnt`
The quantity bought of an item.
Example: `1X` [same row: `@0`]

### 1.4 `menu.unitprice`
The price of one unit of an item.
Example: `35000.00` [same row: `REGULAR`, `2`, `70000.00`]

### 1.5 `menu.price`
The price for the item line as a whole (all units of that item).
Example: `38,000` [same row: `1`, `Onde-Onde`]

### 1.6 `menu.vatyn`
A mark that says whether the item is taxed.
Example: none yet.

## 2. Subtotal

### 2.1 `sub_total.subtotal_price`
The sum of the items before tax and other charges are added.
Example: `SUB TOTAL 53,000`

### 2.2 `sub_total.tax_price`
A tax amount, such as restaurant tax (PB1, P.Rest) or VAT (PPN).
Example: `P.Rest 10 % 5,592`

## 3. Total and payment

### 3.1 `total.total_price`
The final amount the customer has to pay.
Example: `TOTAL 45,500`

### 3.2 `total.cashprice`
The amount paid in cash.
Example: `CASH 50,000`

### 3.3 `total.changeprice`
The change given back to the customer (Indonesian: *kembali*).
Example: `Kembali 27.000`

### 3.4 `total.creditcardprice`
The amount paid by debit or credit card.
Example: `VISA 283,338`

### 3.5 `total.emoneyprice`
The amount paid with an e-wallet or e-money (for example GoPay).
Example: `GoPay 72.000`

### 3.6 `total.menuqty_cnt`
The total quantity of items bought.
Example: `Jumlah Item 1`

### 3.7 `total.menutype_cnt`
The number of different items bought.
Example: `Total Item 7`

## 4. Other labels

These labels exist but have no rule yet:
`menu.discountprice`, `menu.etc`, `menu.itemsubtotal`, `menu.sub.cnt`, `menu.sub.nm`,
`menu.sub.price`, `menu.sub.unitprice`, `sub_total.discount_price`, `sub_total.etc`,
`sub_total.othersvc_price`, `sub_total.service_price`, `total.total_etc`, `void_menu.nm`,
`void_menu.price`.

## Changelog
- v1: initial guideline.
