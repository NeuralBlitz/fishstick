const express = require('express');
const app = express();
const port = 5000;

app.get('/', (req, res) => {
  res.send('<h1>Project Imported Successfully</h1><p>The server is running on Replit.</p>');
});

app.listen(port, '0.0.0.0', () => {
  console.log(`Server listening at http://0.0.0.0:${port}`);
});
