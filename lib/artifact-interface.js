
var Web3 = require('web3')
var web3Utils = require('web3-utils')
//var ethereumjs = require('ethereumjs-tx');
const Tx = require('ethereumjs-tx')

const Vault = require("./vault");
const miningLogger = require("./mining-logger");
const addr = require("./addr");

var tokenContractJSON = require('../contracts/VirtualMiningBoard.json');
const baseArtifactAddress = addr.VRIG; //'0xc491abf6df867990f0544825da7175861f44ec73';

var childContractJSON = require('../contracts/ChildArtifact.json');
const childArtifactAddress = addr.VGPU; //'0x105bf9d11c1dee63297fe2ab113313e54ab18736';

var mineableContractJSON = require('../contracts/MineableToken.json');

var web3 = new Web3();

function truncate0xFromString(s) {
    if (s.startsWith('0x')) {
        return s.substring(2);
    }
    return s;
}

function splitStr(s) {
	var list = s.split(",").map(function(item) {
	  return item.trim();
	});
	return list;
} 

module.exports = {

	init(web3, vault, miningLogger) {
        this.web3 = web3;
        this.artifactContract = new web3.eth.Contract(tokenContractJSON.abi, baseArtifactAddress);
        this.childContract = new web3.eth.Contract(childContractJSON.abi, childArtifactAddress);
        this.mineableContract = new web3.eth.Contract(mineableContractJSON.abi, vault.getTokenContractAddress());
        this.miningLogger = miningLogger;
        this.vault = vault;
    },

    async handleChildCommand(subsystem_command, subsystem_option){
    	if (subsystem_command === 'list') {
    		this.listChildArtifacts();
    	}
    },

    async configArtifacts(artifactId, childArtifactIds) {
	var vgpus = splitStr(childArtifactIds);
        console.log('vgpus: ' + vgpus);
	await this.send( this.artifactContract.methods.configureChildren(artifactId, vgpus), baseArtifactAddress );
    },

    async addChildArtifact(artifactId, childArtifactId) {
	await this.send( this.artifactContract.methods.addChildArtifact(artifactId, childArtifactId), baseArtifactAddress );
    },

    async removeChildArtifact(artifactId, childArtifactId) {
        await this.send( this.artifactContract.methods.removeChildArtifact(artifactId, childArtifactId), baseArtifactAddress );
    },

    async installArtifact(artifactId) {
	await this.send( this.mineableContract.methods.installBooster(artifactId), this.vault.getTokenContractAddress() );
    },

    async uninstallArtifact() {
	await this.send( this.mineableContract.methods.uninstallBooster(), this.vault.getTokenContractAddress() );
    },

    async handleArtifactCommand(subsystem_command, subsystem_option, subsystem_option2) {

    	if (subsystem_command === 'list') {
    	    this.listArtifacts();
    	}

	if (subsystem_command === 'config') {
    	    this.configArtifacts(subsystem_option, subsystem_option2);
    	}

    	if (subsystem_command === 'add') {
            this.addChildArtifact(subsystem_option, subsystem_option2);    
        }

        if (subsystem_command === 'remove') {
            this.removeChildArtifact(subsystem_option, subsystem_option2);    
        }

        if (subsystem_command === 'install') {
            this.installArtifact(subsystem_option);    
        }

        if (subsystem_command === 'uninstall') {
            this.uninstallArtifact();    
        }
    },

    async listChildArtifacts() {
    	let account = await this.vault.getAccount().address;
    	let count = await this.childContract.methods.balanceOf(account).call();
	console.log();
    	for(var j = 0; j < count; j++){
    		let tokenId = await this.childContract.methods.tokenOfOwnerByIndex(account, j).call();
    		let socketArtifact = await this.childContract.methods.artifactAt(tokenId).call();
		this.displaySocketArtifact(socketArtifact, tokenId);
	}
    },

    async listArtifacts() {
    	let account = await this.vault.getAccount().address;
    	let count = await this.artifactContract.methods.balanceOf(account).call();
	console.log();
    	for(var j = 0; j < count; j++){
    		let tokenId = await this.artifactContract.methods.tokenOfOwnerByIndex(account, j).call();
    		let mergedStats = await this.artifactContract.methods.mergedStats(tokenId).call();
    		this.displayBoosterStats(mergedStats, tokenId);
    	}
    },

    async displayBaseStats(booster, id) {
		displayBoosterStats(await booster.baseStats(id), id);
	},

	async displayMergedStats(booster, id) {
		displayBoosterStats(await booster.mergedStats(id), id);
	},

    displayBoosterStats(stats, id) {

		let name = stats[0];
		let basicStats = stats[1];
	  	let ex = basicStats[0];
		let ld = basicStats[1];
	   	let ec = basicStats[2];
	   	let sok = basicStats[3];
	   	let vhash = basicStats[4];
	   	let acc = basicStats[5];
	   	let lvl = basicStats[6];
	   	let childArtifacts = stats[2];

	   	console.log('\n💎 ----------- Virtual Rig ---------- 💎');
	   	console.log('💎\tId: ' + id);
        	console.log('💎\tName: ' + name);
        	console.log('💎\tExperience: ' + ex);
        	console.log('💎\tLife Decrement: ' + ld);
        	console.log('💎\tExecutionCost: ' + ec);
		console.log('💎\tTotal Socket Slots: ' + sok);
		console.log('💎\tvHash: ' + vhash);
		console.log('💎\tAccuracy: ' + acc);		
		console.log('💎\tLevel: ' + lvl);
		console.log('💎\tChildren: ' + childArtifacts);
	  	console.log('💎 ------------------------------------ 💎');

	},

	displaySocketArtifact(stats, id) {
	  	let name = stats[0];
		let parent = stats[1];
		let life = stats[2];
		let modifiers = stats[3];

		console.log('\t💎 ------- Virtual GPU -------- ');
		console.log('\t💎\tName: ' + name);
		console.log('\t💎\tParent: ' + parent);
		console.log('\t💎\tLife: ' + life);
		console.log('\t💎\tModifiers: ');
		for(var j = 0; j < modifiers.length; j++){
			this.displayModifier(modifiers[j]);
		}  
		console.log('\t -------------------------------- ');
		console.log();

	},

	getPositionName(position) {
		if(position == 0){
			return 'Experience';
		}else if (position == 1){
			return 'Life Decrement';
		}else if (position == 2){
			return 'Execution Cost';
		}else if (position == 3){
			return 'Socket Count';
		}else if (position == 4){
			return 'Virtual Hash';
		}else if (position == 5){
			return 'Accuracy';
		}else if (position == 6){
			return 'Level';
		}else{
			return '[' + position + ']';
		}
	},

	parseExponent(op) {
		var s = new String(op);

		var multiplier = s.substring(0,1);
		var exp = s.substring(1,3);

		return new Number(multiplier) + '*10^' + new Number(exp);
	},

	parseCommand(command) {
		var s = new String(command);
		var position = s.substring(1, 3);
		var value = s.substring(3);

		var op = value.substring(0,1);
		var modValue = value.substring(1,4);

        	return [new Number(position), new Number(value), new Number(op), new Number(modValue)];
	},

	displayModifier(modifier){
		var tuple = this.parseCommand(modifier);
		var position = tuple[0];
		var value = tuple[1];
		var op = tuple[2];
		var mod = tuple[3];

		if(op == 1) console.log('\t💎\t[+] Add '+ mod +' to ' + this.getPositionName(position) );
		if(op == 2) console.log('\t💎\t[-] Subtract '+ mod +' from ' + this.getPositionName(position) );
		if(op == 3) console.log('\t💎\t[*] Multiply '+ this.getPositionName(position) +' by ' + mod );
		if(op == 4) console.log('\t💎\t[/] Divide '+ this.getPositionName(position) +' by ' + mod );
		if(op == 5) console.log('\t💎\t[+%] Add '+ mod +'% to ' + this.getPositionName(position) );
		if(op == 6) console.log('\t💎\t[-%] Subtract '+ mod +'% from ' + this.getPositionName(position) );
		if(op == 7) console.log('\t💎\tRequire '+ this.getPositionName(position) +' > ' + mod );
		if(op == 8) console.log('\t💎\tRequire '+ this.getPositionName(position) +' < ' + mod );
		if(op == 9) console.log('\t💎\tAdd ' + this.parseExponent(mod)  + ' to ' + this.getPositionName(position));	

	},

	async send ( method, contractAddress ) {
		let account = await this.vault.getAccount().address;
	    	var fullPrivKey = this.vault.getFullAccount().privateKey;
		var privateKey = truncate0xFromString(fullPrivKey);

		var gasPriceOps = web3Utils.toBN( this.vault.getArtifactOpsGasPriceGwei() );
	    	var data = method.encodeABI();
	    	var nonce = await this.web3.eth.getTransactionCount(account);

		var tx = new Tx({
		      nonce: nonce,
		      gasPrice: web3Utils.toHex(web3Utils.toWei(gasPriceOps, 'gwei')),
		      gasLimit: 500000,
		      to: contractAddress,
		      value: 0,
		      data: data,
		});

		tx.sign(new Buffer(privateKey, 'hex'));

		var raw = '0x' + tx.serialize().toString('hex');
		this.web3.eth.sendSignedTransaction(raw, function (err, transactionHash) {
		    	if(err){
		    		console.log(err);
		    	}else {
		      		console.log('txn: ' + transactionHash);
		  	}
		});
	}

}
