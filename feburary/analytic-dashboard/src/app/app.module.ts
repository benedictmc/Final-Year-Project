import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { GoogleModule } from './google.module';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HomeComponent } from './home/home.component';


import { FusionChartsModule } from 'angular-fusioncharts';
import * as powercharts from 'fusioncharts/fusioncharts.powercharts';
import { ChartsComponent } from './charts/charts.component';

import * as FusionCharts from 'fusioncharts'

import * as Charts from 'fusioncharts/fusioncharts.charts'

import * as FusionTheme from 'fusioncharts/themes/fusioncharts.theme.fusion'

Charts(FusionCharts);

FusionTheme(FusionCharts);

FusionChartsModule.fcRoot(FusionCharts);


@NgModule({
  declarations: [
    AppComponent,
    HomeComponent,
    ChartsComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FusionChartsModule,
    GoogleModule,
    HttpClientModule,
    BrowserAnimationsModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
