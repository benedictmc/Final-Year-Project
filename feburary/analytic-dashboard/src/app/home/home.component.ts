import { Component, OnInit } from '@angular/core';
import { Observable, Subject, ReplaySubject, from, of, range } from 'rxjs';
import { HttpClient } from '@angular/common/http';
import { catchError, tap, map } from 'rxjs/operators';
import {TAdata} from '../class/TAdata';
import { dataClass } from '../class/dataClass';
import { signal }from '../class/signal'


  export interface PeriodicElement {
    time: number;
    position: string;
    newTime: number;
    close: string;
  }
  
  const ELEMENT_DATA: PeriodicElement[] = [
    {time: 1, position: 'Hydrogen', newTime: 1.0079, close: 'H'},
    {time: 2, position: 'Helium', newTime: 4.0026, close: 'He'},
    {time: 3, position: 'Lithium', newTime: 6.941, close: 'Li'},
    {time: 4, position: 'Beryllium', newTime: 9.0122, close: 'Be'},
    {time: 5, position: 'Boron', newTime: 10.811, close: 'B'},
    {time: 6, position: 'Carbon', newTime: 12.0107, close: 'C'},
    {time: 7, position: 'Nitrogen', newTime: 14.0067, close: 'N'},
    {time: 8, position: 'Oxygen', newTime: 15.9994, close: 'O'},
    {time: 9, position: 'Fluorine', newTime: 18.9984, close: 'F'},
    {time: 10, position: 'Neon', newTime: 20.1797, close: 'Ne'},
  ];


@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.styl']
})
export class HomeComponent implements OnInit {
  ipUrl = "http://127.0.0.1:5000/"
  toggleTable = false
  featuresNames = []
  signals = []
  updated: number
  TAmap = new Map<string, number>(); 
  data: dataClass[]
  latestSignal: string
  loaded: boolean = false
  ws_s: WebSocket = new WebSocket("ws://127.0.0.1:5679/")
  ws_d: WebSocket = new WebSocket("ws://127.0.0.1:5678/")

  displayedColumns: string[] = ['time', 'position', 'newTime', 'close'];
  dataSource = ELEMENT_DATA;



  constructor(public http: HttpClient) { }

  ngOnInit() {
    // this.getCoins().subscribe(data =>{
    //   this.coins = data
    //   console.log(this.coins)
    // })

    this.getFeatures().subscribe(data =>{
      console.log('Feature names is')
      this.featuresNames = data
    })

    this.listen_ws_display_data(this.ws_d)
    this.listen_ws_signal(this.ws_s)

    // END POINT NOT WORKING ***
    // this.getSignalData().subscribe(data =>{
    //   console.log("This is the signa; data")
    //   console.log(data['signal'])
    //   this.signals = data['signal']
      
    //   if(this.signals[Object.keys(data['signal'])[0]] == 1)
    //     this.latestSignal = 'Buy'
    //   else
    //     this.latestSignal = 'Sell'
    // })

    // this.getDisplayData().subscribe(data =>{
    //     console.log("This is the display data")
    //     this.data = data
    //     this.loaded = true
    //     this.updated = data[0]['date']
    //   })
  }

  listen_ws_display_data(socket){
    console.log("WS data display")
    let self = this
    socket.addEventListener('message', function (event) {
      let msg = JSON.parse(event.data)
      console.log("Recieving display data")
      self.data = msg.message
      self.updated = msg.message[0]['date']
      self.loaded = true
    });
  }

  listen_ws_signal(socket){
    console.log("WS data signal")
    let self = this
    socket.addEventListener('message', function (event) {
      let msg = JSON.parse(event.data)
      console.log("Recieving signal data")
      self.signals = msg.message
      console.log(msg.message[0])
      console.log(typeof(msg.message))


    });
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



  getDisplayData (): Observable<any[]> {
    let url = this.ipUrl+'API/display-data'

    return this.http.get<dataClass[]>(url)
      .pipe(
        tap(_ => _),
        catchError(this.handleError('getFeatures', []))
      );
  }

  getSignalData (): Observable<any[]> {
    let url = this.ipUrl+'API/signal-data'
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
