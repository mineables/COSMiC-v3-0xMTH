var web3utils = require('web3-utils');
const BN = require('bn.js');
var debugLogger = require('./lib/debug-logger')
const miningLogger = require("./lib/mining-logger");
var tokenContractJSON = require('./contracts/_0xBitcoinToken.json');
var CPPMiner = require('./build/Release/hybridminer');
var donationPercent = 1.5;                                 // Please support the developers! :)
                                                           // Valid Settings: 0 to 70 in increments of 0.5

//only load this if selecting 'gpu mine!!!'

var tokenContract;

const PRINT_STATS_TIMEOUT = 5000;
const COLLECT_MINING_PARAMS_TIMEOUT = 4000;
var hardwareType = 'cuda'; //default

// 2% DevFee - please see !ReadMe_First.txt if you need help changing this
var donationShareCounter = 0;
var donationOffset = 5 + (Math.round(Math.random()*20));    // sol # out of 200 where donated shares start
var sharesDonatedThis200 = 0;                               // shares donated (counts up til donationPercent, resets to 0)
var sharesToDonatePer200 = donationPercent*2;               // 2*donationPercent since this is of a 'batch' of 200 shares

var solutionsSubmitted = 0;

module.exports = {
    async init(web3, vault, miningLogger)
    //  async init(web3, subsystem_command, vault, networkInterface, miningLogger)
    {
        process.on('exit', () => {
            miningLogger.print("Process exiting.\n");
            CPPMiner.stop();
    } );

        tokenContract = new web3.eth.Contract(tokenContractJSON.abi, vault.getTokenContractAddress());

        this.miningLogger = miningLogger;
        this.vault = vault;
    },

    async mine(subsystem_command, subsystem_option) {
        if (subsystem_option == 'cuda') {
            CPPMiner.setHardwareType('cuda');
        } else if (subsystem_option == 'opencl') {
            CPPMiner.setHardwareType('opencl');
        } else {
            CPPMiner.setHardwareType('cpu');
        }

        //console.log('\n')

        //miningParameters

        if (this.miningStyle == "solo")
        {
            //disable auto-donation unless pool mining
            console.log ("♥ Solo Mining: Auto-Donation is disabled.");
            donationPercent = 0;

            //if solo mining need a full account
            var eth_account = this.vault.getFullAccount();

            if (eth_account.accountType == "readOnly" || eth_account.privateKey == null || typeof eth_account.privateKey == 'undefined ')
            {
                miningLogger.print("The account", eth_account.address, 'does not have an associated private key. Please select another account or mine to a pool.');
                //console.log('\n')
                return;
            }
        }
        else if (this.miningStyle == "pool")
        {
            if (donationPercent <= 0)
                console.log ("\n♥ Auto-donation disabled. Please consider supporting this software! See !ReadMe_First.txt.");
            else
            {
                // avoid undefined behavior by limiting donation to 60% max for miners who use the donation feature to split
                // up their earnings. TODO: better workaround!
                if (donationPercent > 60)
                {
                    console.log ("\n♥ Auto-donation: You're too generous! Lowering donation to 70% :)")
                    donationPercent = 70;
                }
                  else
                    console.log ("\n♥ Auto-donation: set to", donationPercent.toFixed(1), "% - Your support is greatly appreciated :)");
            }
        
            // debug only
            //console.log ("donationOffset =", donationOffset);

            var eth_account = this.vault.getAccount();
        }

        if (eth_account == null || eth_account.address == null) {
            miningLogger.print("Please create a new account with 'account new' before solo mining.")
            //console.log('\n')
            return false;
//        } else {
//            miningLogger.print("Selected mining account:\n\t", eth_account.address);
            //console.log('\n')
        }

        ///this.mining = true;
        this.minerEthAddress = eth_account.address;

        let miningParameters = {};
        await this.collectMiningParameters(this.minerEthAddress, miningParameters, this.miningStyle);

        this.miningLogger.appendToStandardLog("Begin mining for " + this.minerEthAddress + " @ gasprice " + this.vault.getGasPriceGwei());

        process.stdout.write('\x1b[s\x1b[?25l\x1b[3;72f\x1b[38;5;33m' + this.minerEthAddress.slice(0, 8) + '\x1b[0m\x1b[u\x1b[?25h');

        if (this.miningStyle != "pool") {
            miningLogger.print("Gas price is", this.vault.getGasPriceGwei(), 'gwei');
        }

        //keep on looping!
        setInterval(async() => { await this.collectMiningParameters(this.minerEthAddress, miningParameters, this.miningStyle) }, COLLECT_MINING_PARAMS_TIMEOUT);

        setInterval(() => { this.printMiningStats() }, PRINT_STATS_TIMEOUT);
    },

    mineStuff(miningParameters) {
        if (!this.mining) {
            this.mineCoins(this.web3, miningParameters, this.minerEthAddress);
        }
    },

    setMiningStyle(style) {
        this.miningStyle = style;
    },

    async collectMiningParameters(minerEthAddress, miningParameters, miningStyle) {
        //    miningLogger.print('collect parameters.. ')
        try {
            if (miningStyle === "pool") {
                var parameters = await this.networkInterface.collectMiningParameters(minerEthAddress, miningParameters);
            } else {
                var parameters = await this.networkInterface.collectMiningParameters();
            }

            //miningLogger.print('collected mining params ', parameters)
            miningParameters.miningDifficulty = parameters.miningDifficulty;
            miningParameters.challengeNumber = parameters.challengeNumber;
            miningParameters.miningTarget = parameters.miningTarget;
            miningParameters.poolEthAddress = parameters.poolEthAddress;

            //give data to the c++ addon
            await this.updateCPUAddonParameters(miningParameters, miningStyle)
        } catch (e) {
            miningLogger.print(e)
        }
    },

    async updateCPUAddonParameters(miningParameters, miningStyle) {
        let bResume = false;

        if (miningStyle == 'pool' && this.challengeNumber != null) {
            //if we are in a pool, keep mining again because our soln probably didnt solve the whole block and we want shares
            //   bResume = true;
            CPPMiner.setChallengeNumber(this.challengeNumber);
            bResume = true;
        }

        if (this.challengeNumber != miningParameters.challengeNumber) {
            this.challengeNumber = miningParameters.challengeNumber

            //miningLogger.print("New challenge received");
            if (solutionsSubmitted)
            //        console.log ("\nInitial challenge received:    ", this.challengeNumber);
            //    else
                console.log ("\nNew challenge received:    ", this.challengeNumber, "\n");

            CPPMiner.setChallengeNumber(this.challengeNumber);
            bResume = true;
            process.stdout.write("\x1b[s\x1b[?25l\x1b[2;13f\x1b[38;5;34m" + this.challengeNumber.substring(2, 10) +
                                 "\x1b[0m\x1b[u\x1b[?25h");
        }

        if (this.miningTarget == null || !this.miningTarget.eq(miningParameters.miningTarget)) {
            this.miningTarget = miningParameters.miningTarget

//            miningLogger.print("New mining target received");
            CPPMiner.setDifficultyTarget("0x" + this.miningTarget.toString(16, 64));
        }

        if (this.miningDifficulty != miningParameters.miningDifficulty) {
            this.miningDifficulty = miningParameters.miningDifficulty

//            miningLogger.print("New difficulty set", this.miningDifficulty);
            process.stdout.write("\x1b[s\x1b[?25l\x1b[3;14f\x1b[38;5;34m" + this.miningDifficulty.toString().padEnd(7) +
                                 "\x1b[0m\x1b[u\x1b[?25h");
//            CPPMiner.setDifficulty( parseInt( this.miningTarget.toString(16, 64).substring(0, 16), 16 ) );
        }

        if (bResume && !this.mining) {
//            miningLogger.print("Restarting mining operations");

            try {
                // C++ module entry point
                this.mineStuff(miningParameters);
            } catch (e) {
                miningLogger.print(e)
            }
        }
    },

    //async submitNewMinedBlock(addressFrom, solution_number, digest_bytes, challenge_number)
    submitNewMinedBlock(addressFrom, minerEthAddress, solution_number, digest_bytes, challenge_number, target, difficulty) {
        //this.miningLogger.appendToStandardLog("Giving mined solution to network interface " + challenge_number);
        if (donationShareCounter > 199)
        {
            donationShareCounter = 0;
            sharesDonatedThis200 = 0;
        }
         else
            ++donationShareCounter;

        var dateStore = new Date();                     // type Date to hold results of function calls below

        // populate the date and include preceding zeroes where applicable
        var hrs = dateStore.getHours();
        hrs = (hrs < 10 ? "0" : "") + hrs;
        var mins  = dateStore.getMinutes();
        mins = (mins < 10 ? "0" : "") + mins;
        var secs  = dateStore.getSeconds();
        secs = (secs < 10 ? "0" : "") + secs;
        //var year = dateStore.getFullYear();
        var month = dateStore.getMonth() + 1;
        month = (month < 10 ? "0" : "") + month;
        var day  = dateStore.getDate();
        day = (day < 10 ? "0" : "") + day;

        // generate the timestamp in square brackets
        var timestamp = "[" + month + "/" + day + " " + hrs + ":" + mins + ":" + secs + "]";

        // Auto-Donation a.k.a. DevFee helps support miner development and will be distributed amongst 0xBTC mining contributors :)
        // Default Devfee: 1.25% (adjust the range below as desired to select frequency w/ which a share is credited to the devfee.
        // All others go to your specified mining account.)
        if (donationShareCounter >= donationOffset && sharesDonatedThis200 < sharesToDonatePer200)
        {
            this.networkInterface.queueMiningSolution(addressFrom, "0xa8b8ea4C083890833f24817b4657888431486444", solution_number, digest_bytes, challenge_number, target, difficulty);
            console.log (timestamp, "Sol'n Found:   ", solution_number, "♥");  //, donationShareCounter,
            ++sharesDonatedThis200;
        }
          else
          {
            this.networkInterface.queueMiningSolution(addressFrom, minerEthAddress, solution_number, digest_bytes, challenge_number, target, difficulty);
            console.log (timestamp, "Sol'n Found:   ", solution_number);       //, donationShareCounter,
          }
    },

    // contractData , -> miningParameters
    mineCoins(web3, miningParameters, minerEthAddress) {
        var target = miningParameters.miningTarget;
        var difficulty = miningParameters.miningDifficulty;

        var addressFrom;

        if (this.miningStyle == "pool") {
            addressFrom = miningParameters.poolEthAddress;
        } else {
            addressFrom = minerEthAddress;
        }

        CPPMiner.setMinerAddress(addressFrom);

        const printSolutionCount = async(solutionString) => {
            process.stdout.write("\x1b[s\x1b[?25l\x1b[3;22f\x1b[38;5;221m" + solutionString +
                                     "\x1b[0m\x1b[u\x1b[?25h");
        }

        const verifyAndSubmit = () => {
            var solution_number = "0x" + CPPMiner.getSolution();
            if(solution_number == "0x" || web3utils.toBN(solution_number).eq(0)) { return; }
            const challenge_number = miningParameters.challengeNumber;
            const digest = web3utils.soliditySha3(challenge_number,
                                                  addressFrom,
                                                  solution_number);
            const digestBigNumber = web3utils.toBN(digest);
            if (digestBigNumber.lte(miningParameters.miningTarget)) {
                solutionsSubmitted++;
//                miningLogger.print("Submitting solution #" + solutionsSubmitted);
                //  this.submitNewMinedBlock(minerEthAddress, solution_number, digest, challenge_number);
                this.submitNewMinedBlock(addressFrom, minerEthAddress, solution_number,
                                         digest, challenge_number, target, difficulty)
                printSolutionCount(solutionsSubmitted.toString().padStart(8));
            //} else {
            //    console.error("Verification failed!\n",
            //                  "challenge:", challenge_number, "\n",
            //                  "address:", addressFrom, "\n",
            //                  "solution:", solution_number, "\n",
            //                  "digest:", digest, "\n",
            //                  "target:", miningParameters.miningTarget);
            }
        }

        setInterval(() => { verifyAndSubmit() }, 500);

        this.mining = true;

        debugLogger.log('MINING:', this.mining)

//        CPPMiner.stop();
        CPPMiner.run((err, sol) => {
            if (sol) {
                try {
                    verifyAndSubmit(sol);
                } catch (e) {
                    miningLogger.print(e)
                }
            }
            this.mining = false;

            debugLogger.log('MINING:', this.mining)
        });
    },

    setHardwareType(type) {
        hardwareType = type;
        miningLogger.print("Set hardware type:", type)
    },

    setNetworkInterface(netInterface) {
        this.networkInterface = netInterface;
    },

    printMiningStats() {
        var hashes = CPPMiner.hashes();
        //  miningLogger.print('hashes:', hashes )
        //miningLogger.print('Hash rate: ' + parseInt(hashes / PRINT_STATS_TIMEOUT) + " kH/s");
    }
}
