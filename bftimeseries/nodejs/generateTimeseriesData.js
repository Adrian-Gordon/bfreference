'use strict'
const fs = require('fs')
const nconf = require('./config/conf.js').nconf
const logger = require('./logger/logger')(module)
const MongoClient=require('mongodb').MongoClient

let gcount = 0
let tcount = 0

let count = gcount

const generateFilepath = nconf.get('generatefilepath')
const testFilepath = nconf.get('testfilepath')
const generateProportion = nconf.get('generateproportion')
const testProportion = nconf.get('testproportion')

const gort = generateProportion / (generateProportion + testProportion)

const gOutStream = fs.createWriteStream(generateFilepath)
const tOutStream = fs.createWriteStream(testFilepath)

MongoClient.connect(nconf.get("databaseurl"),{useNewUrlParser: true},(err,database) => {
 //logger.info("connected");
    if(err) throw(err)

    gOutStream.write("No,layprice1,laydepth1,layprice2,laydepth2,layprice3,laydepth3,layprice4,laydepth4,layprice5,laydepth5,layprice6,laydepth6,layprice7,laydepth7,layprice8,laydepth8,layprice9,laydepth9,layprice10,laydepth10,backprice1,backdepth1,backprice2,backdepth2,backprice3,backdepth3,backprice4,backdepth4,backprice5,backdepth5,backprice6,backdepth6,backprice7,backdepth7,backprice8,backdepth8,backprice9,backdepth9,backprice10,backdepth10\n")
    tOutStream.write("No,layprice1,laydepth1,layprice2,laydepth2,layprice3,laydepth3,layprice4,laydepth4,layprice5,laydepth5,layprice6,laydepth6,layprice7,laydepth7,layprice8,laydepth8,layprice9,laydepth9,layprice10,laydepth10,backprice1,backdepth1,backprice2,backdepth2,backprice3,backdepth3,backprice4,backdepth4,backprice5,backdepth5,backprice6,backdepth6,backprice7,backdepth7,backprice8,backdepth8,backprice9,backdepth9,backprice10,backdepth10\n")
    

    let cursor = database.db('rpdata').collection("timeseries").find()
    //console.log("const dataset = [")
    cursor.forEach(race => {
    //  console.log(race.marketid)
      const marketHistory = race.markethistory
      for(let horseid in marketHistory){

        let stream = gOutStream
       
        if(Math.random() > gort){
          stream = tOutStream
         
        }
        
        const horse = marketHistory[horseid]
    //    console.log(horseid + " " + horse.status)
        if(horse.status == 'ACTIVE'){
          for(let i=0; i<60; i++){

            if(stream == gOutStream){
              gcount++
              count = gcount
            }
            else{
              tcount++
              count = tcount
            }
            //lay
            stream.write(count + "," + horse.layprice1[i] + "," + horse.laydepth1[i] + "," + horse.layprice2[i] + "," + horse.laydepth2[i] + ","  + horse.layprice3[i] + "," + horse.laydepth3[i] + "," + horse.layprice4[i] + "," + horse.laydepth4[i] + ","  + horse.layprice5[i] + "," + horse.laydepth5[i] + ","  + horse.layprice6[i] + "," + horse.laydepth6[i] + ","  + horse.layprice7[i] + "," + horse.laydepth7[i] + ","  + horse.layprice8[i] + "," + horse.laydepth8[i] + ","  + horse.layprice9[i] + "," + horse.laydepth9[i] + ","  + horse.layprice10[i] + "," + horse.laydepth10[i] + "," )
            //back
            stream.write(horse.backprice1[i] + "," + horse.backdepth1[i] + "," + horse.backprice2[i] + "," + horse.backdepth2[i] + ","  + horse.backprice3[i] + "," + horse.backdepth3[i] + "," + horse.backprice4[i] + "," + horse.backdepth4[i] + ","  + horse.backprice5[i] + "," + horse.backdepth5[i] + ","  + horse.backprice6[i] + "," + horse.backdepth6[i] + ","  + horse.backprice7[i] + "," + horse.backdepth7[i] + ","  + horse.backprice8[i] + "," + horse.backdepth8[i] + ","  + horse.backprice9[i] + "," + horse.backdepth9[i] + ","  + horse.backprice10[i] + "," + horse.backdepth10[i])
            stream.write("\n")
            
          }


        }

      }
    })
   // gOutStream.end()
    //tOutStream.end()
    //process.exit(0)
  })