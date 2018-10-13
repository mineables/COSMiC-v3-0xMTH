var MINING_POOL_URL = 'http://pool.mineables.io'

var web3Utils = require('web3-utils')
const Tx = require('ethereumjs-tx')
const Vault = require("./vault");
const miningLogger = require("./mining-logger")
var jayson = require('jayson');
const fetch = require('node-fetch')
var tokenContractJSON = require('../contracts/MineableToken.json')

var busySendingSolution = false;
var queuedMiningSolutions = [];
var lastSubmittedMiningSolutionChallengeNumber;

const MAX_TARGET = web3Utils.toBN( 2 ).pow( web3Utils.toBN( 234 ) )

module.exports = {
    init(web3, subsystem_command, vault, miningLogger) {
        this.web3 = web3;
	this.tokenContractAddress = vault.getTokenContractAddress()
        this.tokenContract = new web3.eth.Contract(tokenContractJSON.abi, vault.getTokenContractAddress())
        this.miningLogger = miningLogger;
        this.vault = vault;
        busySendingSolution = false;

        if (this.vault.getMiningPool() == null) {
            this.vault.selectMiningPool(MINING_POOL_URL)
        }

        this.jsonrpcClient = jayson.client.http(
            this.vault.getMiningPool()
        );

        setInterval(async () => { await this.sendMiningSolutions() }, 500)
    },

    async handlePoolCommand(subsystem_command, subsystem_option) {
        if (subsystem_command === 'select') {
            this.vault.selectMiningPool(subsystem_option); //pool url
            await this.vault.saveVaultData();
        }

        if (subsystem_command === 'show' || subsystem_command === 'list') {
            miningLogger.print('Selected mining pool:', this.vault.getMiningPool())
        }
    },

    //the miner will ask for this info to help find solutions !!
    hasReceivedPoolConfig() {
        return this.receivedPoolConfig;
    },

    getPoolEthAddress() {
        return this.poolEthAddress;
    },

    getMinimumShareDifficulty() {
        return this.poolMinimumShareDifficulty;
    },

    targetFromDifficulty(difficulty) {
      return MAX_TARGET.div( web3Utils.toBN(difficulty) )
    },

    async collectMiningParameters(minerEthAddress, previousMiningParameters) {
        var challengeNumber = await this.tokenContract.methods.getChallengeNumber().call()

        if(challengeNumber === previousMiningParameters.challengeNumber) {
            // console.log('challengeNumber is the same...')
            return previousMiningParameters
        } else {
            var shareRequest = {}
            shareRequest.origin = minerEthAddress
            shareRequest.contract = this.tokenContractAddress
	    shareRequest.vardiff = this.vault.getVardiff()
            let response = await (await fetch(MINING_POOL_URL + '/share/request', 
                                               { 
                                                 method: 'POST', 
                                                 body: JSON.stringify(shareRequest), 
                                                 headers: { 'Content-Type': 'application/json' } 
                                               })).json()

	    this.currentUuid = response._id

	    // console.log(response)
            return {
                miningDifficulty: response.difficulty,
                challengeNumber: response.challengeNumber,
                miningTarget: this.targetFromDifficulty(response.difficulty),
                poolEthAddress: minerEthAddress
            }
        }
    },

    async sendMiningSolutions() {
        //  miningLogger.print( 'sendMiningSolutions' )
        if (busySendingSolution == false) {
            if (queuedMiningSolutions.length > 0)
            {
                //busySendingSolution = true;
                var nextSolution = queuedMiningSolutions.pop();

                this.miningLogger.appendToStandardLog("Sending queued solution", nextSolution.toString())
                console.log (" ==> Sent sol'n to pool.");

                //in the pool miner we send the next soln to the pool regardless

                //  if( nextSolution.challenge_number != lastSubmittedMiningSolutionChallengeNumber)
                //  {
                //  lastSubmittedMiningSolutionChallengeNumber =  nextSolution.challenge_number;

                try
                {
                    var response = await this.submitMiningSolution(nextSolution.addressFrom, nextSolution.minerEthAddress,
                                                                   nextSolution.solution_number, nextSolution.challenge_number,
                                                                   nextSolution.challenge_digest, nextSolution.target,
                                                                   nextSolution.difficulty);
                }
                catch (e)
                {
                    this.miningLogger.appendToErrorLog(e)
                    miningLogger.print(e);
                }
                //    }
                busySendingSolution = false;
            }
        }
    },

    async queueMiningSolution(addressFrom, minerEthAddress, solution_number, challenge_digest, challenge_number, target, difficulty) {
        //miningLogger.print('pushed solution to stack')
        queuedMiningSolutions.push({
            addressFrom: addressFrom, //the pool in the pools case,  the miner if solo mining
            minerEthAddress: minerEthAddress, // ALWAYS miner eth address
            solution_number: solution_number,
            challenge_digest: challenge_digest,
            challenge_number: challenge_number,
            target: target,
            difficulty: difficulty
        });
    },

    async submitMiningSolution(addressFrom, minerEthAddress, solution_number, challenge_number, challenge_digest, target, difficulty) {

        var acct = this.vault.getAccount()
        var submitShare = this.prepareDelegatedMintTxn(solution_number, acct)
        submitShare.uid = this.currentUuid

        let res = await fetch(MINING_POOL_URL + '/share/submit', { method: 'POST', 
                                                                   body: JSON.stringify(submitShare), 
                                                                   headers: { 'Content-Type': 'application/json' } })

	// now set up the next around
        var shareRequest = {}
        shareRequest.origin = minerEthAddress
        shareRequest.contract = this.tokenContractAddress
	shareRequest.vardiff = this.vault.getVardiff()
        let response = await (await fetch(MINING_POOL_URL + '/share/request', 
                                         { 
                                           method: 'POST', 
                                           body: JSON.stringify(shareRequest), 
                                           headers: { 'Content-Type': 'application/json' } 
                                         })).json()
	this.currentUuid = response._id

    },

    prepareDelegatedMintTxn(nonce) {
	  var functionSig = web3Utils.sha3("delegatedMintHashing(uint256,address)").substring(0,10)
	  var data = web3Utils.soliditySha3( functionSig, nonce, this.vault.getFullAccount().address )
	  var sig = this.web3.eth.accounts.sign(web3Utils.toHex(data), this.vault.getFullAccount().privateKey )
	  // prepare the mint packet
	  var packet = {}
	  packet.nonce = nonce
	  packet.origin = this.vault.getFullAccount().address
	  packet.signature = sig.signature
	  // deliver resulting JSON packet to pool or third party
	  var mineableMintPacket = JSON.stringify(packet, null, 4)
	  /* todo: send mineableMintPacket to submitter */
	  return packet
    },
    async sendSignedRawTransaction(web3, txOptions, addressFrom, vault, callback) {

        var fullPrivKey = vault.getAccount().privateKey;
        var privKey = this.truncate0xFromString(fullPrivKey)

        const privateKey = new Buffer(privKey, 'hex')
        const transaction = new Tx(txOptions)

        transaction.sign(privateKey)

        const serializedTx = transaction.serialize().toString('hex')

        try {
            var result = web3.eth.sendSignedTransaction('0x' + serializedTx, callback)
        } catch (e) {
            miningLogger.print(e);
        }
    },

    truncate0xFromString(s) {
        if (s.startsWith('0x')) {
            return s.substring(2);
        }
        return s;
    }
}
