var googleTrends = require("google-trends-api")


// googleTrends.interestOverTime({keyword: 'nebulas coin', startTime: new Date('2018-10-21'), geo: 'US', granularTimeResolution: true}).then(function(results){
//   console.log(results);
// })
// .catch(function(err){
//   console.error(err);
// });



const rows = [["name1", "city1", "some other info"], ["name2", "city2", "more info"]];
let csvContent = "data:text/csv;charset=utf-8,";
rows.forEach(function(rowArray){
   let row = rowArray.join(",");
   csvContent += row + "\r\n";
}); 

var encodedUri = encodeURI(csvContent);