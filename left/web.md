# 渲染

```
下载html, css, js文件
执行 JavaScript
HTML 文档构建/ JS 更新 DOM 树
CSS 文件构建/ JS 更新 CSSOM 树
DOM + CSSOM → 渲染树（Render Tree）
    ↓
布局（Layout） → 计算每个渲染树节点在视口中的确切位置和尺寸
    ↓
绘制（Paint） → 像素化内容
    ↓
合成（Composite）
```

每个现代浏览器都内置一个 JavaScript 引擎，负责解析和执行 JS 代码

JS 通过浏览器提供的 Web API 来实现与页面、用户、网络的交互

要运行 JavaScript，本质上需要两样东西：

JavaScript 引擎

宿主环境:

​	浏览器：提供 document、fetch、setTimeout 等 Web API；

​    Node.js：提供 fs（文件系统）、http、path 等系统 API。

React / Vue / Angular 是运行在浏览器中的 JavaScript 库/框架，用于构建复杂的 Web 应用界面，但最终仍依赖浏览器原生能力来渲染和交互。

```
Skia 是一个让不同操作系统和硬件都能高效绘制 2D 图形（“画一个圆”、“渲染文字”）的通用中间层
   ├─ CPU 模式：生成位图（Bitmap）  
   └─ GPU 模式：通过OpenGL/Vulkan/Metal，生成纹理（Texture）
一块包含最终可视内容的像素缓冲区——要么在 CPU 内存，要么在 GPU 显存

App-level Compositing
输入：多个图层纹理 + 变换矩阵；
执行者：GPU（通过 OpenGL/Vulkan/Metal）
输出：一帧最终像素，提交给操作系统

System-level Compositing
合成多个应用窗口 + 桌面元素
```

```
Firefox 在 Linux 上仅对系统级对话框（文件选择器、打印对话框等）调用 GTK，而核心 UI（标签页/地址栏）仍由自研引擎绘制。
GNOME 是 GTK 应用，KDE 是 Qt 应用
Qt 是一个功能完整的跨平台 C++ 应用开发框架，涵盖 UI、网络、文件、线程、数据库、多媒体等几乎所有应用开发所需模块。在桌面专业软件领域占据不可替代的地位，但不是手机 App 的首选。有自己的一套跨平台图形抽象层，不是Skia，但底层也是OpenGL/Vulkan/Metal
任何桌面环境均可运行任意框架的应用，只要采用相同的协议
```

# 扩展

浏览器扩展（Browser Extension）是一种运行在浏览器内部、用于增强或修改浏览器功能的小型软件。利用浏览器提供的扩展 API，在特定时机向网页或浏览器界面注入代码，从而实现自定义功能。

现代浏览器扩展（如 Chrome / Edge / Firefox 的 WebExtensions）采用 多进程/多上下文架构

内容脚本（Content Script）—— 在指定网页中运行 JS/CSS，修改页面内容或行为。和网页 JS 都能操作 DOM，但它们运行在不同的 JavaScript 执行上下文（execution context）中，彼此隔离。

后台页（Background Page）—— 监听浏览器事件（如标签页切换、网络请求），执行长期任务；

油猴插件（Tampermonkey）是一种流行的用户脚本管理器（User Script Manager），它允许用户在浏览器中自动运行自定义 JavaScript 代码，以修改网页行为或外观。它的核心原理是在网页加载过程中注入用户脚本，绕过了隔离

# 油猴脚本

浏览器使用开发人员模式，油猴允许访问文件 URL

f12，切换到 Elements 面板，在页面上找到一个“可预订”的时间段按钮，右键点击该按钮，选择 “Copy → Copy selector”

脚本不能使用需要重启浏览器

```
// ==UserScript==
// @name         羽毛球场自动预订
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  刷新 → 点日期 → 点时段 → 点确认
// @author       You
// @match        https://gym.sysu.edu.cn/*
// @grant        none
// ==/UserScript==

(function() {
	'use strict';

	const TARGET_HOUR = 21;
    const TARGET_MIN = 59;//57
    const TARGET_SEC = 59;
    const TARGET_MS = 670;//650太早

    const TARGET_DATE_INDEX = 4;
    //const TARGET_DATE_INDEX = 2;
    const TARGET_ROWS = [19, 20];
    //const TARGET_ROWS = [15, 16];
    //const TARGET_ROWS = [8, 9];
    const TARGET_COLS = [4, 3, 2, 8, 9, 7, 10, 6, 5, 1];
    //const TARGET_COLS = [4, 2, 1];

	const tip = document.createElement('div');
    tip.id = 'booking-tip';
    tip.style.cssText = `
        position: fixed; top: 20px; right: 20px;
        background: #000; color: #fff; padding: 10px;
        border-radius: 6px; z-index: 999999;
	`;
    document.body.appendChild(tip);
    function log(msg) { tip.textContent = msg; }

    function clickDate() {
        const sel = `#app > div > div.main-container > main > div > div.content-grid > div.venue-selection > div.date-selector > div:nth-child(${TARGET_DATE_INDEX})`;
        const el = document.querySelector(sel);
        return el && (el.click(), true);
    }

    function clickConfirm() {
        const btn = document.querySelector('#app > div > div.main-container > div.booking-footer > div > button');
        return btn && (btn.click(), true);
    }

    function tryBookSlot(row, col) {
        let adjRow = row;
        if (row>11) {
            adjRow = row - 9;
        }else {
            adjRow = row - 7;
        }
        const adjCol = col + 1;
        const selector = `#app > div > div.main-container > main > div > div.content-grid > div.venue-selection > div.time-slot-table > table > tbody > tr:nth-child(${adjRow}) > td:nth-child(${adjCol}) > button`;
        const btn = document.querySelector(selector);
        if (btn && !btn.disabled && !btn.classList.contains('sold')) {
            btn.click();
            return true;
        }
        return false;
    }

	async function bookAllPossible() {
        while (1) {
            if (clickDate()) {
                break;
            }
            await new Promise(r => setTimeout(r, 1));
        }
        log(`A`);
        const adjRow = 2;
        const adjCol = 2;
        while (1) {
            const selector = `#app > div > div.main-container > main > div > div.content-grid > div.venue-selection > div.time-slot-table > table > tbody > tr:nth-child(${adjRow}) > td:nth-child(${adjCol}) > button`;
            const btn = document.querySelector(selector);
            if (btn) {
                break;
            }
            await new Promise(r => setTimeout(r, 1));
        }
        await new Promise(r => setTimeout(r, 10));
        log(`B`);
        let count = 0;
        for (let round = 0; round < 1; round++) {
            for (let row of TARGET_ROWS) {
                for (let col of TARGET_COLS) {
                    await new Promise(r => setTimeout(r, 1));
                    if (tryBookSlot(row, col)) {
                        count = count + 1;
                        break;
                    }
                }
            }
        }
        log(`C${count}`);
        while (1) {
            if (clickConfirm()) {
                break;
            }
            await new Promise(r => setTimeout(r, 1));
        }
        log(`D${count}`);
    }

    if (new URLSearchParams(window.location.search).has('auto')) {
        //const now = new Date();
        //log(`${now.getSeconds()},${now.getMilliseconds()}`);
        //21.722-20.700=1s
        log(`0`);
        bookAllPossible();
        return;
    }

    const timer = setInterval(() => {
        const now = new Date();
        if (
            (now.getHours() === TARGET_HOUR &&
            now.getMinutes() === TARGET_MIN &&
            now.getSeconds() === TARGET_SEC &&
            now.getMilliseconds() >= TARGET_MS) ||
            (now.getHours() === TARGET_HOUR+1)
        ) {
            clearInterval(timer);
            window.location.search = '?auto=1';
        }
        log(`${now.getSeconds()},${now.getMilliseconds()/100}`);
    }, 1);
})();
```

```
// ==UserScript==
// @name         羽毛球场自动预订
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  刷新 → 点日期 → 点时段 → 点确认
// @author       You
// @match        https://gym.sysu.edu.cn/*
// @grant        none
// ==/UserScript==

(function() {
	'use strict';

	const TARGET_HOUR = 21;
    const TARGET_MIN = 59;
    const TARGET_SEC = 59;
    const TARGET_MS = 550;
    //const TARGET_HOUR = 22;
    //const TARGET_MIN = 2;
    //const TARGET_SEC = 10;

    const TARGET_DATE_INDEX = 4;
    //const TARGET_DATE_INDEX = 2;
    const TARGET_ROWS = [19, 20];
    //const TARGET_ROWS = [15, 16];
    //const TARGET_ROWS = [8, 9];
    const TARGET_COLS = [8, 9, 7, 4, 3, 2, 10, 6, 5, 1];
    //const TARGET_COLS = [2, 1];

	const tip = document.createElement('div');
    tip.id = 'booking-tip';
    tip.style.cssText = `
        position: fixed; top: 20px; right: 20px;
        background: #000; color: #fff; padding: 10px;
        border-radius: 6px; z-index: 999999;
	`;
    document.body.appendChild(tip);
    function log(msg) { tip.textContent = '📌 ' + msg; }

    function clickDate() {
        const sel = `#app > div > div.main-container > main > div > div.content-grid > div.venue-selection > div.date-selector > div:nth-child(${TARGET_DATE_INDEX})`;
        const el = document.querySelector(sel);
        return el && (el.click(), true);
    }

    function clickConfirm() {
        const btn = document.querySelector('#app > div > div.main-container > div.booking-footer > div > button');
        return btn && (btn.click(), true);
    }

    function tryBookSlot(row, col) {
        let adjRow = row;
        if (row>11) {
            adjRow = row - 9;
        }else {
            adjRow = row - 7;
        }
        const adjCol = col + 1;
        const selector = `#app > div > div.main-container > main > div > div.content-grid > div.venue-selection > div.time-slot-table > table > tbody > tr:nth-child(${adjRow}) > td:nth-child(${adjCol}) > button`;
        const btn = document.querySelector(selector);
        if (btn && !btn.disabled && !btn.classList.contains('sold')) {
            btn.click();
            return true;
        }
        return false;
    }

	async function bookAllPossible() {
        while (1) {
            if (clickDate()) {
                break;
            }
            await new Promise(r => setTimeout(r, 1));
        }
        await new Promise(r => setTimeout(r, 1));
        //wait new date table
        log(`A`);
        let count = 0;
        for (let round = 0; round < 1; round++) {
            for (let row of TARGET_ROWS) {
                for (let col of TARGET_COLS) {
                    if (tryBookSlot(row, col)) {
                        break;
                        //await new Promise(r => setTimeout(r, 1));
                        ////wait confirm button
                        //if (clickConfirm()) {
                        //    await new Promise(r => setTimeout(r, 150));
                        //    //you must wait to book the second one successfully
                        //    log(`${row} 点, ${col} 号已预订`);
                        //    break;
                        //}
                    }
                }
            }
        }
        log(`B`);
        log(`C${count}`);

        await new Promise(r => setTimeout(r, 1));
        //wait confirm button
        if (clickConfirm()) {
            log(`Confirm`);
        }else{
            log(`not Confirmed`);
        }
        log(`D${count}`);

    }

    if (new URLSearchParams(window.location.search).has('auto')) {
        //const now = new Date();
        //log(`${now.getSeconds()},${now.getMilliseconds()}`);
        //21.722-20.700=1s
        setTimeout(bookAllPossible, 220);
        return;
    }

    const timer = setInterval(() => {
        const now = new Date();
        if (
            (now.getHours() === TARGET_HOUR &&
            now.getMinutes() === TARGET_MIN &&
            now.getSeconds() === TARGET_SEC &&
            now.getMilliseconds() >= TARGET_MS) ||
            (now.getHours() === TARGET_HOUR+1)
        ) {
            clearInterval(timer);
            log('刷新页面以加载新日期...');
            window.location.search = '?auto=1';
        }
        log(`${now.getSeconds()},${now.getMilliseconds()/100}`);
    }, 1);
})();
```



# bookmarklets

点击该书签时，Bookmarklet 会在网页的 JavaScript 执行上下文中运行

刷新 = 加载全新页面 = 旧 JS 环境被销毁

不能有注释  

```
javascript:(function(){
	alert('hello world');
})();
```

