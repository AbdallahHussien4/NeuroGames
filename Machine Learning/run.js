// import package
const WebSocket = require('ws')

// create websocket server
const wss = new WebSocket.Server({ host: "192.168.137.1", port: 8080 }, () => {
    console.log('server started')
})

// websocket server is established
wss.on('listening', () => {
    console.log('listening on 8080')
})

// Initialize a variable that would contain the model's prediction (the action to be taken).
var action = 0;

// Run the streaming python script as a process.
var spawn = require('child_process').spawn,
    ls    = spawn('python',['stream.py']);

// Fetch the streaming output from stdout
ls.stdout.on('data', function (data) {
    action = data.toString()[0]
    console.log('stdout: ' + data);
});

// On encountering an error set the action to 0 so that it is neutral.
ls.stderr.on('data', function (data) {
    action = 0
    console.log('stderr: ' + data);
});

// A new connection to the server
wss.on('connection', function connection(ws) {

    console.log('A new connection !!')

    // send data to the client every time limit
    setInterval(() => {
        ws.send(action);
        console.log('data snt \n %o', action)
    }, 2000)

    // received message from client
    ws.on('message', (data) => {
        console.log('data received \n %o', data)
        ws.send(data);
    })

    // a client disconnected
    ws.on('close', (code, reason) => {
        console.log('connection ' + code + "closed")
    })

    // a new connection is established
    ws.on('open', () => {
        console.log('A new open connection !!')
    })
})