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
  constructor(public http: HttpClient) { }

  ngOnInit() {
    this.getCoins().subscribe(data =>{
      this.coins = data
    })

    this.getFeatures().subscribe(data =>{
      this.features = data
    })

    this.getFeatureData().subscribe(data =>{
      this.featureData = data
      this.TAdata = <TAdata>this.featureData.indicators
      type TAdsata = typeof TAdata;
      const headers: Array<Object> = Object.keys(TAdata).map(key => {
          return { text: key, value: key }
      });    })
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



  private handleError<T> (operation = 'operation', result?: T) {
    return (error: any): Observable<T> => {
      console.error(error); // log to console instead
      console.log(`${operation} failed: ${error.message}`);
      return of(result as T);
    };
  }
}
