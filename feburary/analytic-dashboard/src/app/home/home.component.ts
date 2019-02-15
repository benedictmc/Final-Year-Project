import { Component, OnInit } from '@angular/core';
import { Observable, Subject, ReplaySubject, from, of, range } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { catchError, map, tap } from 'rxjs/operators';
import {TAdata} from '../class/TAdata';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.styl']
})
export class HomeComponent implements OnInit {
  ipUrl = "http://127.0.0.1:5000/"
  coins = []
  features = []
  featureData = []
  TAdata: TAdata
  TAmap = new Map<string, number>(); 

  constructor(public http: HttpClient) { }

  ngOnInit() {
    this.getCoins().subscribe(data =>{
      this.coins = data
      console.log(this.coins)
    })

    this.getFeatures().subscribe(data =>{
      this.features = data
    })

    this.getFeatureData().subscribe(data =>{
      this.featureData = data
      console.log("This is the feature data")

      console.log(this.featureData)
      // this.TAdata = <TAdata>this.featureData.indicators
      // this.mapTAData(this.TAdata)
      })

    this.getTestData().subscribe(data =>{
        console.log("This is the test data")
        console.log(data)
      })
  }

  mapTAData(taData){
    this.features.forEach(element => {
      this.TAmap.set(element,taData[element])
    });
    console.log(this.TAmap)
  }

  getCoins (): Observable<any[]> {
    let url = this.ipUrl+'API/coins'
    console.log(url)

    return this.http.get<any[]>(url)
      .pipe(
        tap(_ => _),
        catchError(this.handleError('getCoins', []))
      );
  }

  getFeatures (): Observable<any[]> {
    let url = this.ipUrl+'API/features'
    console.log(url)

    return this.http.get<any[]>(url)
      .pipe(
        tap(_ => _),
        catchError(this.handleError('getFeatures', []))
      );
  }

  getFeatureData (): Observable<any[]> {
    let url = this.ipUrl+'API/feature-data'
    console.log(url)

    return this.http.get<any[]>(url)
      .pipe(
        tap(_ => _),
        catchError(this.handleError('getFeatures', []))
      );
  }

  getTestData (): Observable<any[]> {
    let url = this.ipUrl+'API/test'

    return this.http.get<any[]>(url)
      .pipe(
        tap(_ => _),
        catchError(this.handleError('getFeatures', []))
      );
  }



  private handleError<T> (operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {
      console.error(error); // log to console instead
      console.log(`${operation} failed: ${error.message}`);
      return of(result as T);
    };
  }
}
