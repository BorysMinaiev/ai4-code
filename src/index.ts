
const OFFSET = 30;
const R = 15;
var lastClosestId = "";
var lastChosenId = "";

function findClosestId(mouseX: number, mouseY: number) {
    var column = 0;
    if (mouseX >= OFFSET - R && mouseX <= OFFSET + R) {
        column = 1;
    }
    const width = get_elem('canvas').width;
    const height = get_elem('canvas').height;
    const rightPos = width - OFFSET;
    if (mouseX >= rightPos - R && mouseX <= rightPos + R) {
        column = 2;
    }
    if (column == 0) {
        return "";
    }
    const nb_id = get_current_nb_id();
    if (nb_id == "") {
        return "";
    }
    const correct_order = correctOrders[nb_id];
    if (correct_order == null) {
        return "";
    }
    const my_order = get_my_order();
    if (my_order.length == 0) {
        return "";
    }

    const len = correct_order.length;
    const delta = (height - OFFSET * 2) / len;
    var expected_pos = Math.round((mouseY - OFFSET) / delta);
    if (expected_pos < 0) {
        expected_pos = 0;
    }
    if (expected_pos >= len) {
        expected_pos = len - 1;
    }
    if (column == 1) {
        return correct_order[expected_pos];
    } else {
        return my_order[expected_pos];
    }
}

class DrawingApp {
    private canvas: HTMLCanvasElement;
    private context: CanvasRenderingContext2D;

    constructor() {
        let canvas = document.getElementById('canvas') as
            HTMLCanvasElement;
        let context = canvas.getContext("2d");
        context.lineCap = 'round';
        context.lineJoin = 'round';
        context.strokeStyle = 'black';
        context.lineWidth = 1;

        this.canvas = canvas;
        this.context = context;

        // this.redraw();
        this.redraw2();
        this.createUserEvents();
    }

    private createUserEvents() {
        let canvas = this.canvas;

        canvas.addEventListener("mousedown", this.pressEventHandler);
        canvas.addEventListener("mousemove", this.dragEventHandler);
        canvas.addEventListener("mouseup", this.releaseEventHandler);
        canvas.addEventListener("mouseout", this.cancelEventHandler);

        canvas.addEventListener("touchstart", this.pressEventHandler);
        canvas.addEventListener("touchmove", this.dragEventHandler);
        canvas.addEventListener("touchend", this.releaseEventHandler);
        canvas.addEventListener("touchcancel", this.cancelEventHandler);

        document.getElementById('clear')
            .addEventListener("click", this.clearEventHandler);
    }


    public redraw2() {
        this.clearCanvas();
        const nb_id = get_current_nb_id();
        if (nb_id == "") {
            return;
        }
        const correct_order = correctOrders[nb_id];
        if (correct_order == null) {
            return;
        }
        const my_order = get_my_order();
        if (my_order.length == 0) {
            return;
        }
        const height = this.context.canvas.height;
        const width = this.context.canvas.width;

        function get_y(pos: number) {
            return OFFSET + (height - OFFSET * 2) * pos / correct_order.length;
        }

        const ctx = this.context;
        for (let i = 0; i < correct_order.length; i++) {
            const y1 = get_y(i);
            const my_pos = my_order.indexOf(correct_order[i]);
            if (my_pos == null) {
                continue;
            }
            const y2 = get_y(my_pos);

            ctx.beginPath();
            ctx.moveTo(OFFSET, y1);
            ctx.lineTo(width - OFFSET, y2);
            if (correct_order[i] == lastChosenId) {
                ctx.strokeStyle = '#0000ff';
                ctx.lineWidth = 2;
            } else
                if (correct_order[i] == lastClosestId) {
                    ctx.strokeStyle = '#ff0000';
                    ctx.lineWidth = 2;
                } else {
                    ctx.strokeStyle = '#000000';
                    ctx.lineWidth = 1;
                }
            ctx.stroke();
            ctx.closePath();

        }
    }


    private pressEventHandler = (e: MouseEvent | TouchEvent) => {
        let mouseX = (e as TouchEvent).changedTouches ?
            (e as TouchEvent).changedTouches[0].pageX :
            (e as MouseEvent).pageX;
        let mouseY = (e as TouchEvent).changedTouches ?
            (e as TouchEvent).changedTouches[0].pageY :
            (e as MouseEvent).pageY;
        mouseX -= this.canvas.offsetLeft;
        mouseY -= this.canvas.offsetTop;

        this.addClick(mouseX, mouseY, false);
    }

    private dragEventHandler = (e: MouseEvent | TouchEvent) => {
        let mouseX = (e as TouchEvent).changedTouches ?
            (e as TouchEvent).changedTouches[0].pageX :
            (e as MouseEvent).pageX;
        let mouseY = (e as TouchEvent).changedTouches ?
            (e as TouchEvent).changedTouches[0].pageY :
            (e as MouseEvent).pageY;
        mouseX -= this.canvas.offsetLeft;
        mouseY -= this.canvas.offsetTop;

        const closestId = findClosestId(mouseX, mouseY);
        if (closestId != lastClosestId) {
            lastClosestId = closestId;
            this.redraw2();
        }

        e.preventDefault();
    }

    private clearEventHandler = () => {
        this.clearCanvas();
    }

    private releaseEventHandler = () => {

    }

    private cancelEventHandler = () => {

    }

    private addClick(x: number, y: number, dragging: boolean) {
        lastChosenId = lastClosestId;
        this.redraw2();
    }

    private clearCanvas() {
        this.context
            .clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
}

const app = new DrawingApp();

import { parse } from 'papaparse';

interface HashTable<T> {
    [key: string]: T;
}

var correctOrders: HashTable<string[]> = {};

function save_local(key: string, value: string) {
    if (value == "") {
        window.localStorage.removeItem(key);
    } else {
        window.localStorage.setItem(key, value);
    }
}

function get_local(key: string) {
    return window.localStorage.getItem(key);
}

function get_elem(id: string) {
    const e = document.getElementById(id);
    if (e == null) {
        return null;
    }
    return (<HTMLInputElement>e);
}

const fetchOrders = async () => {
    const response = await fetch('http://localhost:8000/AI4Code/train_orders.csv');
    const data = await response.text();

    parse<Array<string>>(data, {
        complete: (result) => {
            const data = result.data;
            for (let i = 1; i < data.length - 1; i++) {
                const item = data[i];
                correctOrders[item[0]] = item[1].split(" ");
            }
            console.log("done!");
            app.redraw2();
        }
    })

}

const NB_ID = "nb_id";
const MY_ORDER = "my_order";

function get_order_key(nb_id: string) {
    return MY_ORDER + "_" + nb_id;
}

function setChangeHandlerNbId() {
    const elem = get_elem(NB_ID);
    elem.value = get_local(NB_ID);

    function reload_order() {
        get_elem(MY_ORDER).value = get_local(get_order_key(elem.value));
        app.redraw2();
    }

    reload_order();
    elem.onkeyup = (_e: KeyboardEvent) => {
        save_local(NB_ID, elem.value);
        reload_order();
    };
}

function get_current_nb_id() {
    const e = get_elem(NB_ID);
    if (e == null) {
        return "";
    }
    return e.value;
}

function get_my_order() {
    const e = get_elem(MY_ORDER);
    if (e == null) {
        return [];
    }
    return e.value.split(",");
}

function setChangeHandlerMyOrder() {
    const elem = get_elem(MY_ORDER);
    elem.onkeyup = (_e: KeyboardEvent) => {
        save_local(get_order_key(get_current_nb_id()), elem.value);
        app.redraw2();
    };
}

function setInputHandlers() {
    setChangeHandlerNbId();
    setChangeHandlerMyOrder();
};

setInputHandlers();
fetchOrders();