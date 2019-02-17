import { Component, OnInit } from '@angular/core';


const data = {
  "chart": {
    "caption": "Recommended Portfolio Split",
    "subcaption": "For a net-worth of $1M",
    "showvalues": "1",
    "showpercentintooltip": "0",
    "numberprefix": "$",
    "enablemultislicing": "1",
    "theme": "fusion"
  },
  "data": [
    {
      "label": "Buy",
      "value": "300000"
    },
    {
      "label": "Sell",
      "value": "230000"
    }
  ]
};

@Component({
  selector: 'app-charts',
  templateUrl: './charts.component.html',
  styleUrls: ['./charts.component.styl']
})
export class ChartsComponent implements OnInit {

    dataSource = {
      "chart": {
        "caption": "Buy/Sell Split 1 Day",
        "subCaption" : "",
        "showValues":"1",
        "showPercentInTooltip" : "0",
        "numberPrefix" : "$",
        "enableMultiSlicing":"1",
        "theme": "fusion"
      },
      "data": [
        {
          "label": "Buy",
          "value": "300000"
        },
        {
          "label": "Sell",
          "value": "230000"
        }
      ]
    }

  constructor() { }
}