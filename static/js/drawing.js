const BACKGROUND_COLOR = '#000000'
const LINE_COLOR = '#FFFFFF'
const LINE_WIDTH = 15

var curX = 0
var cury = 0
var prevX = 0
var prevy = 0

var c
var ctx

function prepareCanvas() {

   // console.log("working")

   c = document.getElementById('my_canvas')
   ctx = c.getContext('2d')

   // style the canvas
   ctx.fillStyle = BACKGROUND_COLOR
   ctx.fillRect(0,0, c.clientWidth,c.clientHeight)

   ctx.strokeStyle = LINE_COLOR
   ctx.lineWidth = LINE_WIDTH
   ctx.lineJoin = 'round'


   var ispainting = false

   // mouse listener
   document.addEventListener('mousedown', function(e){
      // console.log('mouse pressed')
      ispainting = true
      curX = e.clientX - c.offsetLeft
      cury = e.clientY - c.offsetTop
   })

   document.addEventListener('mousemove', function(e){
      if (ispainting){
         prevX = curX
         curX = e.clientX - c.offsetLeft // we subtract it cause we want origin from canvas

         prevy = cury
         cury = e.clientY - c.offsetTop

         draw()
      }

   })

   document.addEventListener('mouseup', function(){
      // console.log('mouse released')
      ispainting = false
   })

   c.addEventListener('mouseleave',function(){
      ispainting = false
   })
}

function draw(){
   ctx.beginPath()
   ctx.moveTo(prevX,prevy)
   ctx.lineTo(curX,cury)
   ctx.closePath()
   ctx.stroke()
}

function cleanCanvas(){
   curX = 0
   cury = 0
   prevX = 0
   prevy = 0
   ctx.fillRect(0,0, c.clientWidth,c.clientHeight)
}
