const fetch = require('node-fetch')

module.exports = {
	async addresses(){
		let res = await fetch('https://mineables.io/static/scripts/config.js')
		let t = await res.text()
		let jsonRes = t.replace(/module.exports = /gi, '')
		let config = JSON.parse(jsonRes)
		console.log(config)
		return config.addresses
	}
}