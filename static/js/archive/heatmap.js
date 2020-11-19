var myMap = L.map("map", {
  center: [ 37.09, -95.71 ],
  zoom: 4.5
});

L.tileLayer("https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
  attribution: "© <a href='https://www.mapbox.com/about/maps/'>Mapbox</a> © <a href='http://www.openstreetmap.org/copyright'>OpenStreetMap</a> <strong><a href='https://www.mapbox.com/map-feedback/' target='_blank'>Improve this map</a></strong>",
  tileSize: 512,
  maxZoom: 18,
  zoomOffset: -1,
  id: "mapbox/streets-v11",
  accessToken: API_KEY
}).addTo(myMap);

var data = "http://127.0.0.1:5000/heatmapdata";

console.log('hello');

d3.json(data, function(data) {

  var heatArray = [];

  for (var i = 0; i < data.length; i++) {

    var latitude = parseFloat(data[i].latitude);
    var longitude = parseFloat(data[i].longitude);

      heatArray.push([latitude, longitude]);
    }

  var heat = L.heatLayer(heatArray, {
    radius: 75,
    blur: 35
  }).addTo(myMap);

});
