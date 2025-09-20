import type { ComponentCommand } from '@commands';
import { i18n } from '@locale';
import { useStore } from '@nanostores/preact';
import type { FunctionalComponent } from 'preact';

const messages = i18n('volunteering', {
	title: 'Volunteering Experience',
	mentor1: 'Mentor - Hack IT Sapiens',
	mentor1Date: 'Apr 2023',
	mentor1Location: 'India',
	mentor1Desc: 'Mentored participants in Hack IT Sapiens event, providing guidance on technical challenges and helping develop solutions in coding and problem-solving. Offered both technical and strategic assistance.',
	mentor2: 'Mentor - Shooting Stars Foundation',
	mentor2Date: 'Dec 2022',
	mentor2Location: 'India',
	mentor2Desc: 'Mentored students in hackathons, guiding them through technical concepts and project execution. Worked on coding, robotics, and web development projects, collaborating with other mentors.',
	secretary: 'AICTE PIET IDEA Lab Secretary',
	secretaryDate: 'Aug 2022 - Dec 2022',
	secretaryLocation: 'Jaipur',
	secretaryDesc: 'Managed administrative and operational tasks for AICTE IDEA Lab, ensuring smooth day-to-day functioning. Responsibilities included event organization, budget management, and project implementation support.',
});

const Volunteering: FunctionalComponent = () => {
	const t = useStore(messages);

	return (
		<div className='terminal-line-history'>
			<h3>{t.title}</h3>
			
			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.mentor1}</strong>
				</p>
				<p>
					{t.mentor1Location} | {t.mentor1Date}
				</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.mentor1Desc}</p>
			</div>

			<div style={{ marginBottom: '1.5rem' }}>
				<p>
					<strong>{t.mentor2}</strong>
				</p>
				<p>
					{t.mentor2Location} | {t.mentor2Date}
				</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.mentor2Desc}</p>
			</div>

			<div style={{ marginBottom: '1rem' }}>
				<p>
					<strong>{t.secretary}</strong>
				</p>
				<p>
					{t.secretaryLocation} | {t.secretaryDate}
				</p>
				<p style={{ color: 'var(--color-text-200)', marginTop: '0.5rem' }}>{t.secretaryDesc}</p>
			</div>
		</div>
	);
};

const VolunteeringCommand: ComponentCommand = {
	command: 'volunteering',
	component: Volunteering,
};

export default VolunteeringCommand;
