const fetch = require('node-fetch');

// Test the basic profile route
async function testProfileRoute() {
  try {
    console.log('Testing profile test route...');
    const response = await fetch('http://localhost:5000/api/profile/test');
    const data = await response.json();
    console.log('Response:', data);
  } catch (error) {
    console.error('Error testing profile route:', error);
  }
}

// Run the tests
testProfileRoute(); 