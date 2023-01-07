using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class crossPunch : StateMachineBehaviour
{
    public healthBar healthbar;
    public AudioClip hit;
    public Transform vrPlayer;
    bool takeDamage;
    AudioSource source;
    bool getShot = false;

    // OnStateEnter is called when a transition starts and the state machine starts to evaluate this state
    override public void OnStateEnter(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        // getting the variables from game objects
        healthbar = GameObject.FindWithTag("PlayerHealth").GetComponent<healthBar>();
        source = GameObject.FindWithTag("fighter").GetComponent<fighter>().GetComponent<AudioSource>();
        vrPlayer = GameObject.FindWithTag("MainCamera").GetComponent<Transform>();

        // initilizing flags
        takeDamage = true;
        getShot = false;
    }

    // OnStateUpdate is called on each Update frame between OnStateEnter and OnStateExit callbacks
    override public void OnStateUpdate(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        // check if user has already taken damage and synchronize the animation
        if(animator.GetCurrentAnimatorStateInfo(0).normalizedTime > 0.65f && !getShot)
        {
            // check if user has dodged the punch
            if(takeDamage)
            {
                source.PlayOneShot(hit);
                healthbar.takeDamage(15);
            }

            // user took damage
            getShot = true;
        }

        // check Z angles of VR player to dodge the punch
        if(!((vrPlayer.eulerAngles.z >= 0 && vrPlayer.eulerAngles.z <= 20) || (vrPlayer.eulerAngles.z >= 340 && vrPlayer.eulerAngles.z <= 360)))
        {
            takeDamage = false;
        }
    }

    // OnStateExit is called when a transition ends and the state machine finishes evaluating this state
    override public void OnStateExit(Animator animator, AnimatorStateInfo stateInfo, int layerIndex)
    {
        getShot = false;
        takeDamage = true;
    }
    
}
